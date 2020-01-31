# feat_from_pandas
## Routines to create and execute feat files (via FSL's flameo OLS) from pandas dataframes.
#### Requires: pandas, os, glob, numpy, and dominate (for HTML generation; conda install dominate)

    Parameters/properties:
        columns   (list; req)  : List of columns from which to create feat files
        data      (list; req)  : Pandas dataframe from which to pull columns of data
        basename  (str;  opt)  : Stem for named feat .mat .con and .grp files.
                                 Default = 'design'
        outputdir (str;  opt)  : Directory path to save feat files.
                                 Default = Current working directory
        image_col (str ; opt)  : Column comtaining paths to participant images of interest
                                 Default = None
        gmv       (bool; opt)  : Add a GMV to design.mat file?
                                 Default = False
        demean    (bool; opt)  : Demean columns in design.mat file?
                                 Default = True
        round_to  (int;  opt)  : Round values in design mat to <n> decimal places.
                                 Default = 3
        ffill     (bool; opt)  : Forward fill missing values in design.mat?
                                 Default = False
        res_fldr  (str ; opt)  : Name of folder w/in outdir to store flameo results. 
                                 Defaults = 'results'. 
        mask_image(str ; opt)  : Full path to mask image when running flameo and easythresh.
                                 Will default to MNI152 mask distributed with FSL.

    Functions:
        show_parameters()      : Print out summary of models and parameters
        save_paramaters()      : Write  summary of models and parameters to outputdir
        write_feats()          : Create and write con, grp and mat files to outputdir
        selfupdate()           : Update/create con, grp and mat representations using the current
                                 parameters, and save string representation in object variable
        make_filtered_func()   : Create 4D nifti image for feat model; 'image_col' must contain
                                 full paths to image for each subject/row.
        run_feat_files()       : Run flameo OLS in outputdir. Feat files, filtered_func_data, 
                                 and mask image must already exist.
        make_files_and_run()   : Create all feat files, filtered_func_data, and mask in 'outputdir',
                                 run flameo OLS, easythresh zstats, and make webpage to review
                                 thresholded zstats. All content saved in 'outputdir'.

    Utility Functions:
        mk_outputdir()         : Check for outputdir, and create if it does not exist
        mk_designcon()         : Create con string representation and save in object variable
        mk_designgrp()         : Create grp string representation and save in object variable
        mk_designmat()         : Create mat string representation and save in object variable
        mk_summary()           : Create string representations of con, mat, grp files and save in object
        cp_mask_file()         : Copy mask file into 'outputdir'. If no file is specified in 'mask_image',
                                 fsl's MNI152_T1_brain_mask will be used.
        thresh_zstats()        : Apply easythresh to zstat images in outputdir/results/ using 'zthresh'
                                 and 'pthresh' values.
        mk_webpage_for_zstats(): Create simple webpage for thresholded zstats. Assumes zstats already 
                                 created and easythresh applied. 

    Example:
        ffp = FeatFromPandas(
                            cytokine_cols,
                            dataframe,
                            outputdir='/projects/biomarkers/cytokine_reho',
                            demean = True,
                            round_to = 3,
                            image_col = 'reho_images',
                            mask_image = '/projects/biomarkers/group_mask.nii.gz')

        ffp.make_files_and_run()

    Note: 
        Utility Functions are mostly for internal use. You can/shoud accomplish what you need
        to do with the standard Functions. You can also create the FeatFileMaker object as above, save/run
        the analysis, and update the columns and outputdir to leverage the same dataframe in a series of 
        analyses. 
        e.g.
        ffp.columns=['age','sex','IL6']
        ffp.outputdir='/projects/biomakers/IL6_reho'
        ffp.make_files_and_run()
        
