Analysis tasks
****************

warp.FiducialFitWarp
---------------------

Description: Aligns image stacks by fitting fiducial spots to Gaussians.

Parameters:

* write\_fiducial\_images -- Flag indicating whether the aligned fiducial images should be saved. These images are helpful for visually verifying the quality of the image alignment.
* initial\_sigma -- initial spot Gaussian standard deviation for the spot  fitting algorithm.
* intensity\_threshold -- minimum spot intensity for spot fitting.
* significance\_threshold --  minimum fit significance for spot fitting.

warp.FiducialCorrelationWarp
-----------------------------

Description: Aligns image stacks by maximizing the cross-correlation between fiducial images. 

Parameters:

* write\_fiducial\_images -- Flag indicating whether the aligned fiducial images should be saved. These images are helpful for visually verifying the quality of the image alignment.

preprocess.DeconvolutionPreprocess
-----------------------------------

Description: High-pass filters and deconvolves the image data in preparation for bit-calling.

Parameters:

* warp\_task -- The name of the warp task that provides the aligned image stacks.
* highpass\_pass -- The standard deviation to use for the high pass filter.
* decon\_sigma -- The standard deviation to use for the Lucy-Richardson deconvolution.
* decon\_iterations -- The number of Lucy-Richardson deconvolution iterations to perform on each image.
* decon\_filter\_size -- The size of the gaussian filter to use for the deconvolution. It is not recommended to set this parameter manually.

optimize.Optimize
------------------

Description: Determines the optimal per-bit scale factors for barcode decoding.

Parameters:

* iteration\_count -- The number of iterations to perform for the optimization.
* fov\_per\_iteration -- The number of fields of view to decode in each round of optimization.
* estimate\_initial\_scale\_factors\_from\_cdf -- Flag indicating if the initial scale factors should be estimated from the pixel intensity cdf. If false, the initial scale factors are all set to 1. If true, the initial scale factors are based on the 90th percentile of the pixe intensity cdf.
* area\_threshold -- The minimum barcode area for barcodes to be used in the calculation of the scale factors.

decode.Decode
---------------

Description: Extract barcodes from all field of views using a pixel-based decoding algorithm. This decoding strategy was originally presented in Moffitt*, Hao*, Wang*, et al, PNAS, 2016.

Parameters:

* crop\_width -- The number of pixels from each edge of the image to exclude from decoding. 
* write_decoded\_images -- Flag indicating if the decoded and intensity images should be written.
* minimum\_area -- The area threshold, below which decoded barcodes are ignored.
* lowpass\_sigma -- The standard deviation for the low pass filter prior to decoding.

filterbarcodes.FilterBarcodes
------------------------------

Description: Filters the decoded barcodes based on area and intensity. This filtering strategy was originally presented in Moffitt*, Hao*, Wang*, et al, PNAS, 2016.

Parameters:

* area\_threshold -- Barcodes with areas below the specified area\_threshold are removed.
* intensity\_threshold -- Barcodes with intensities below this threshold are removed.

filterbarcodes.GenerateAdaptiveThreshold
-------------------------------------------

Description: Generate the barcode parameter histograms for the mean intensity, minimimum distance, and area for filtering barcodes with an adaptive threshold. This is run concurrently with decoding in order to minimize the required time.

Parameters:

* run\_after\_task -- The task to start generating the adaptive threshold after. To run concurrently with decode, this can be specified as the preprocess task, otherwise it can be specified as the decode task.
* tolerance -- Tolerance from zero in the optimization routine, if left unset defaults to 0.001. It is useful to adjust this to be more tolerant if an experiment has few blanks or few barcodes overall, as it can be impossible to hit within 0.1% of the requested misidentification rate.

filterbarcodes.AdaptiveFilterBarcodes
----------------------------------------

Description: Use an adaptive barcode to enrich the decode barcodes for the correct barcodes. The adaptive filter selects bins from the three dimension mean intensity, minimum distance, area histogram based on the fraction of blanks within each bin in order to achieve a specified misidentification rate. This filtering strategy was originally presented in Xia*, Fan*, Emanuel*, et al, PNAS, 2019.

Parameters:

* misidentification_rate -- The target misidentification rate, calculated as the number of blank barcodes per blank barcode divided by the number of coding barcodes per coding barcode.

segment.SegmentCells
----------------------

Description: Determines cell boundaries using a watershed algorithm with the seeds determined from a nuclear stain and the watershed performed on a cell stain.

Parameters:

* seed\_channel\_name -- The name of the data channel to use to find seeds
* watershed\_channel\_name -- The name of the data channel to use as the watershed image.W

segment.CleanCellBoundaries
--------------------------------

Description: Assigns each cell to the FOV centroid they are closest to, and eliminates overlapping cells, preferentially removing cells that overlap with the largest number of other cells until there is no more overlap in a given group of cells.

segment.RefineCellDatabases
--------------------------------

Description: Creates a new cell database based on an initial cell database and a set of cells to keep.

segment.ExportCellMetadata
--------------------------------

Description: Exports a csv containing the cell metadata, i.e. fov, volume, x and y coordinates.

generatemosaic.GenerateMosaic
-------------------------------

Description: Assembles the images from each field of view into a low resolution mosaic.

Parameters:

* microns\_per\_pixel -- The number of microns to correspond with a pixel in the mosaic. If set to "full_resolution", the mosaic is generated with the same resolution as the input images.
* data\_channels -- The names of the data channels to export, corresponding to the data organization. If not provided, all data channels are exported.
* z\_indexes -- The z index to export. If not provided all z indexes are exported.
* fov\_crop\_width -- The number of pixels to remove from each edge of each fov before inserting it into the mosaic.
* draw\_fov\_labels -- Flag indicating if the fov index should be drawn on top of each fov in the mosaic
sequential.SumSignal
-------------------------------

Description: Calculates the total intensity within segementation boundaries.

Parameters:

* z\_index -- the z index of the image stack to use for the summation
* apply_highpass -- flag indicating if a highpass filter should be applied to the image prior to summing.
* highpass\_sigma -- the standard deviation to use for the high pass filter

sequential.ExportSumSignals
----------------------------------

Description: Export the sum signals calculated by a SumSignal task to a csv file.

partition.PartitionBarcodes
-------------------------------

Description: Assigns RNAs to cells if the RNA falls within the segmentation boundary of the cell. Yields a counts per cell csv file for a given fov.

partition.ExportPartitionedBarcodes
----------------------------------

Description: Combines the counts per cell csv files from each fov into a single output file.

slurmreport.SlurmReport
-------------------------------

Description: An analysis task that generates reports on previously completed analysis tasks using Slurm. This analysis task only works when Merlin is run through Slurm with every analysis task fragment run as a separate job. This task uploads the Slurm report to a central repository to track Merlin's performance.

Parameters:

* run\_after\_task -- the task to wait for before generating the Slurm report

plotperformance.PlotPerformance
-------------------------------

Description: Create quality control plots of the analysis tasks as soon as the analysis results become avaliable.

Parameters:

* decode\_task 
* filter\_task
* optimize\_task
* segment\_task
* sum\_task
* partition\_task  
* global\_align\_task  
