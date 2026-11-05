Notes:
- Each slide should be 1920 × 1080 pixels with a rectangle border
- The url contains the current page, so reloading the page doesn't bring us back to the beginning
- Have navigation buttons (next/prev) always at the bottom of each page, and let the arrow keys (left/right) and home/end also control this. This is an interactive slide presentation.
- The next and prev buttons should have a single word hint of the content.
- On the main page, we should have a crafted by human link that takes us to the https://magland.github.io/arewehuman/ app with the appropriate data embedded in the url for the crafting of the recipe.md.

## Title slide:

Presentation title: Figpack information and planning session

Jeremy Magland
Center for Computational Mathematics, Flatiron Institute

5 November 2026

## Why Figpack?

Left panel:

Static figures do not carry nearly as much information as interactive web applications backed by large datasets.

Idea: The web application is the figure.

Right panel top:
A static figure. 4 Channels of ephys data with some low frequency modulation and spikes. Traces should be colored by channel.

Right panel bottom:
An interactive web application showing the same data (same four channels), but with controls to explore more in depth. Click to set the current time point. Pan/zoom with the mouse as well as with toolbar controls. Lazy loading of data. Zoomed in to the same window as the static figure, but user can zoom out.

Note that the web application is not technically going to be a figpack figure. It's to get across the idea that exploring data with controls and lazy loading is more powerful than just sharing a static figure.

## Figpack figure example

Left panel:

Create from Python (or MATLAB)

Show a syntax-highlighted python script that generates sample 4-channel time series data (30 seconds at 20 kHz), and creates a TimeseriesGraph figpack figure (see the figpack documentation).

The upload=True parameter should be circled

Right panel:
The generated figure embedded as iframe (run the script and generate the figure, and then I will pin it and provide the URL). 

Below the iframe, show the URL linking to the figure in a new tab.

https://figures.figpack.org/figures/default/595aafbd9ac7a0b3e580d1ad/index.html

## Synchronized views

Left panel:

Views in a figure share state:
* Current time and visible time range (time series, spectrograms, videos)
* Selected units (spike sorting views)

Indicate where the data come from (Allen Brain Observatory, DANDI:000728).

Right panel:

Pinned figure: Allen two-photon session with the visual stimulus video, the imaging movie, dF/F traces and running speed, all synchronized.

https://figures.figpack.org/figures/default/13e864a656937f972da6cd05/index.html

## Exporting as SVG

Left pane:

Export as SVG button
* Export the figure as SVG (vector graphics) for inclusion in publication or presentation
* UI controls and other features are not shown
* Layout is redrawn based on user-selected dimensions for the output
* Not all views support it (ask Jeremy to expand support)

Right pane:

An example pinned figure that has timeseries graphs and other views in a more complex layout (views that support export to SVG)

## A figure is a self-contained directory

Left panel:

A core principle of Figpack is that figures are always self-contained directories.

* Portable
* Shareable
* Long-term archivable
* No external dependencies

Right panel:

Show the directory/file structure of a figure. Include index.html, the javascript, the data

## Efficient data layout - consolidated Zarr

Left panel:

Although not a strict requirement, figure data is generally laid out using a variant of the cloud-friendly Zarr format.

* N-dimensional arrays stored in chunks
* Chunks are compressed
* Supports lazy loading
* Multi-scale storage (zoom out without loading entire dataset)
* Consolidated chunks so total number of files is small (unique to Figpack)

Right panel:

Show a graphical depiction of the layout, including consolidated chunk files.

## Where does the figure live, and for how long?

Left panel:

By default figures expire in 24 hours. Rarely deleted though. The purpose is so there is no expectation that the figure will be available long term (different from figurl).

**Unless the figure is renewed or pinned.**

Multi-tier:
* Ephemeral (no API key required) - cannot be renewed
* Default - expires in 24 hours
* Renewed - expires in an additional 7 days
* Pinned - does not expire, but won't be around forever
* Pinned in your own bucket - will survive as long as you maintain the bucket
* Hosted on GitHub Pages
* Archived on a service like Zenodo (long term - can accompany publication)

Right panel:

Diagram showing this multi-tier, with short descriptions (e.g., single words)

## Managing a figure

Iframe pointing to the management page for the above pinned figure

https://figures.figpack.org/figures/default/595aafbd9ac7a0b3e580d1ad/index.html

## Figpack Gallery

Left panel:

A selection of figures made for this presentation (listed in app/gallery/README.md), scrollable, each one has a name

Somehow indicate that the full gallery is found on the documentation page, with a link.

Right panel:

Depending on which was selected on the left, an iframe pointing to the selected figure

## Integrations

Show (with icons for each) that figpack is integrated into

SpikeInterface

DataJoint

Spyglass (future; right now it uses figurl)

## What if the view I need isn't available?

Left panel:

If you can imagine it, it's probably possible!

With coding agents... implementation is usually straightforward.

Your options:
* Request the feature/view on the figpack GitHub repo.
* Ask Jeremy directly via email/Slack
* Implement it yourself. (Point a coding agent to the codebase)

Right panel:

Diagram showing the different options

## New experimental feature - REMIX

Left panel:

Suppose you created a large figure, and you want to adjust colors, layout, labels, etc, or even add new views of the underlying data.

Select the "REMIX" button on the figure management page.

Chat with AI to make your one-off changes, specific to that figure.

Nice for preparing a one-off figure for a publication or presentation.

Right panel:

Show an example remix interface - use the pinned figure above.

## Remix flow

Show a diagram that illustrates the flow

Start with a figure.

Optionally create a copy (required if figure is owned by someone else).

Remix using AI chat.

Save remix (modifies the figure)

Or export patch (to apply at figure curation time to future figures)

Or apply patch (from previous exported patch)

## New Feature - Annotations

Iframe pointing to the pinned example figure with annotation enabled (TimeseriesGraph with allow_annotation=True). Annotate time intervals and export them as CSV.

https://figures.figpack.org/figures/default/1e88f4bfde208fd89f4e2d1f/index.html

## Possible future directions

* Adding new view types (need input from users)
* Building more advanced annotation/curation capabilities (building into workflow)
* Make it easier to archive figures (host on GitHub Pages or Zenodo)
* What else? ...