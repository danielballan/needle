import numpy as np


def video(fname, mimetype):
    """Load the video in the file `fname`, with given mimetype, and display as HTML5 video.
    """
    from IPython.display import HTML
    video_encoded = open(fname, "rb").read().encode("base64")

    video_tag = """<video controls>
<source alt="test" src="data:video/{0};base64,{1}" type="video/webm">
Use Google Chrome browser.</video>""".format(mimetype, video_encoded)
    return HTML(data=video_tag)


def annotate(video_frames, trajectory, indexer=slice(None, None, None)):
    """Show a video tracing the wire center and orientation.

    Parameters
    ----------
    video_frames : iterable of image arrays
    trajectory : DataFrame
        indexed by frame number, including columns ['x', 'y', 'angle']
    indexer : slice
        slice object like slice(1, 5, 2) specifiying a subset of the video

    Returns
    -------
    IPython display HTML output
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as manimation
    import tempfile
    from IPython.display import display


    FFMpegWriter = manimation.writers['ffmpeg']
    writer = FFMpegWriter(fps=3, extra_args=['-vcodec', 'libvpx'])
    dpi = 100  # dots per inch; controls video size

    v = video_frames  # just a shorthand

    # Create dummy plot elements, to be filled with actual data below.
    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros(v.frame_shape), interpolation='none',
                   vmin=np.iinfo(v.pixel_type).min,
                   vmax=np.iinfo(v.pixel_type).max)
    dot, = ax.plot([], [], 'ro')
    line, = ax.plot([], [], 'r-')
    x_data = np.array([0, v.frame_shape[1]])

    with tempfile.NamedTemporaryFile(suffix='.webm') as temp:
        with writer.saving(fig, temp.name, dpi):
            for frame in v[indexer]:
                # Update the data in the image to the current frame.
                im.set_array(frame)
                # Update the data backing the point and line.
                frame_no = frame.frame_no
                x, y, angle = trajectory.loc[frame_no, ['x', 'y', 'angle']]
                m = np.tan(angle)
                b = y - m*x
                dot.set_data(x, y)
                line.set_data(x_data, m*x_data + b)
            
                ax.set(xlim=(0,v.frame_shape[0]), ylim=(0, v.frame_shape[1]))
                writer.grab_frame()
        temp.flush()
        fig.clf()
        display(video(temp.name, 'x-webm'))


def plot_principal_axes(img, x_bar, y_bar, cov, ax=None):
    """Plot bars with a length of 2 stddev along the principal axes.

    Attribution
    -----------
    This function is based on a solution by Joe Kington, posted on Stack
    Overflow at http://stackoverflow.com/questions/5869891/
    how-to-calculate-the-axis-of-orientation/5873296#5873296
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    def make_lines(eigvals, eigvecs, mean, i):
        """Make lines a length of 2 stddev."""
        std = np.sqrt(eigvals[i])
        vec = 2 * std * eigvecs[:,i] / np.hypot(*eigvecs[:,i])
        x, y = np.vstack((mean-vec, mean, mean+vec)).T
        return x, y
    mean = np.array([x_bar, y_bar])
    eigvals, eigvecs = np.linalg.eigh(cov)
    ax.plot(*make_lines(eigvals, eigvecs, mean, 0), marker='o', color='white')
    ax.plot(*make_lines(eigvals, eigvecs, mean, -1), marker='o', color='red')
    ax.imshow(img)
