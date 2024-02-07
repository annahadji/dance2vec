"""Helper functions to load data."""
import pathlib
import yaml

import numpy as np
import pandas as pd
import cv2

from . import features

FONT = cv2.FONT_HERSHEY_SIMPLEX


def get_sample_video_height(video_path: str) -> int:
    """Return the frame height of video in path."""
    cap = cv2.VideoCapture(video_path)
    return cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


def get_video_from_labels_path(labels_path: pathlib.Path) -> str:
    """Return video str name from labels path."""
    video_path = str(labels_path.parent).replace("labeled-data", "videos")
    video = video_path.split("/")[-1].replace(
        video_path.split("/")[-1], f"{video_path.split('/')[-1]}.mp4"
    )
    return str(pathlib.Path(video_path).parent / video)


def transform_y(row: np.ndarray, frame_height: int = 500) -> np.ndarray:
    """Reverse the y coordinate of a data point to increase from bottom left
    instead of top left (as is the default for image recording)."""
    return frame_height - row


def get_crop_params_for_video_from_config(video_path: str, config_path: str) -> tuple:
    """Return crop parameters from config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    x1, x2, y1, y2 = [
        vals["crop"]
        for video, vals in config["video_sets"].items()
        if video_path in video
    ][0].split(", ")
    return int(x1), int(x2), int(y1), int(y2)


def add_labels_to_frame(
    video_path: str, img: np.ndarray, labels_df: pd.DataFrame
) -> np.ndarray:
    """Add labels to a numpy image, indicated by a circle with text label above."""
    # Load config file for correcting for original cropping when labelling
    config_path = str(pathlib.Path(video_path).parents[1] / "config.yaml")
    x1, _, y1, _ = get_crop_params_for_video_from_config(video_path, config_path)
    # Annotate frame with individuals coordinates
    individuals = set(i[0] for i in labels_df.index)
    for ind in individuals:
        df_ind = labels_df.xs(ind).astype(float).T
        x = df_ind["x"].values[0].astype(int) + int(x1)
        y = df_ind["y"].values[0].astype(int) + int(y1)
        img = cv2.circle(img, (x, y), 10, (255, 255, 0), thickness=-1)
        cv2.putText(img, ind, (x - 35, y - 25), FONT, 1, (225, 255, 0), 2)
    return img


def create_labelled_video(video_path: str, df: pd.DataFrame) -> None:
    """Create a labelled mp4 video from a video and a dataframe of labels."""
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    save_path = video_path.replace(".mp4", "_labelled.mp4").replace(
        "/videos/", "/videos/annotated/"
    )
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(
        save_path,
        cv2.VideoWriter_fourcc(*"mp4v"),  # Define codec
        25,  # FPS
        (width, height),
    )
    # Iterate through frames and add labels
    for frame in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, img = cap.read()
        if ret:
            cv2.putText(img, f"{frame}", (35, 50), FONT, 1, (225, 255, 0), 2)
            # Get coordinates for labels in frame
            label_coords_for_frame = (
                df.iloc[:, df.columns.get_level_values(0) == frame]
                .dropna()
                .xs("back", level="bodypart")
            )
            if label_coords_for_frame.empty is False:
                img = add_labels_to_frame(video_path, img, label_coords_for_frame)
            out.write(img)
    # Release everything on job finish
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def df_from_manual_labels(data_path: pathlib.Path) -> pd.DataFrame:
    """Load and organise manually annotated data."""
    print(data_path)
    df = pd.read_csv(data_path, skiprows=1, low_memory=False)
    # Change frame filename to frame number
    frames = (
        df[df.columns[2]]
        .iloc[2:]
        .apply(lambda row: int(row.split("img")[1].split(".")[0]))
        .values
    )
    df = df.drop(labels=[df.columns[1], df.columns[2]], axis=1).T.reset_index()
    df = df.rename(columns={"index": "individual", 0: "bodypart", 1: "coords"})
    # Pandas loads in as n1, n1.1, n1.2, etc. for one individual
    df["individual"] = df.apply(lambda row: row["individual"].split(".")[0], axis=1)
    # Drop first row of data (old header)
    df = df.iloc[1:, :]
    # Set correct index
    df = df.set_index(["individual", "bodypart", "coords"])
    # Shift timeseries columns backwards to adjust for previous shift due to indiv / bodypt / coords
    df.columns = frames
    # Drop frames and individuals we have not yet labelled
    df = df.dropna(how="all", axis=1)  # Frames
    # Create labelled video with annotations of the bees in the frames
    video_path = get_video_from_labels_path(data_path)
    # create_labelled_video(video_path, df)  # Outputs a video with labels overlaid
    # Y axis increases from top left; change this to increase from bottom right
    frame_height = get_sample_video_height(video_path)  # Used to reverse y axis order
    idx = pd.IndexSlice
    df.loc[idx[:, :, "y"], :] = df.xs("y", level="coords").apply(
        lambda row: transform_y(row.astype(float), frame_height), axis=1
    )
    return df


def load_error_results_for_bees(files, nbins: int):
    """Returns three datasets: the error of nestmate's vector computed for
    all timesteps, the mean error averaged across all timesteps, and the error computed
    for bins of angles to the dancer (irrespective of bee id), and the successive errors
    computed for individuals over consecutive waggle phases."""
    # Load error results for each bee
    assert len(files) > 0, "No files found."
    errors, bees, angles_to_dancer = [], [], []
    for i in files:
        errors_np_file = np.load(str(i))
        bee_id = str(i).split("/")[-1].split("errors_")[1].split(".npz")[0]
        # Error decoded from hdelta memory on each timestep
        errors.append(errors_np_file["errors"])  # Radians
        angles_to_dancer.append(errors_np_file["angle_to_dancer"])  # Radians
        bees.append(bee_id)
    errors_df = pd.DataFrame(
        {
            "bee_id": bees,
            "vector_error_rad": np.array(errors, dtype=object),
            "angle_to_dancer_rad": np.array(angles_to_dancer, dtype=object),
        }
    ).explode(["vector_error_rad", "angle_to_dancer_rad"])
    # Correct data types
    errors_df["vector_error_rad"] = errors_df["vector_error_rad"].astype(np.float64)
    errors_df["vector_error_deg"] = np.degrees(
        errors_df["vector_error_rad"].astype(np.float64)
    )
    errors_df["angle_to_dancer_rad"] = errors_df["angle_to_dancer_rad"].astype(
        np.float64
    )
    errors_df["angle_to_dancer_deg"] = np.degrees(
        errors_df["angle_to_dancer_rad"].astype(np.float64)
    )
    errors_df["frame"] = errors_df.groupby("bee_id").cumcount()  # Add frame number

    # Compute mean error for each bee, and bins
    reduced_df, binned_df = features.reduce_statistics_for_errors(errors_df, nbins)
    # Add mean for consecutive waggle runs for the same bee.
    # Update the estimate of the mean vector error
    consec_waggle_phases = features.compute_mean_across_waggle_runs(errors_df)
    return errors_df, reduced_df, binned_df, consec_waggle_phases
