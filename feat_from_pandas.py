import numpy as np
import os
import pandas as pd


class FeatFromPandas:
    """Routines to create feat files (e.g. for use in flameo) from pandas dataframes.

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
                            outputdir='/projects/stan/biomarkers/cytokine_reho',
                            demean = True,
                            round_to = 3,
                            image_col = 'reho_images',
                            mask_image = '/projects/stan/biomarkers/group_mask.nii.gz')

        ffp.make_files_and_run()

    Note: 
        Utility Functions are mostly for internal use. You can/shoud accomplish what you need
        to do with the standard Functions. You can also create the FeatFileMaker object as above, save/run
        the analysis, and update the columns and outputdir to leverage the same dataframe in a series of 
        analyses. 
        e.g.
        ffp.columns=['age','sex','IL6']
        ffp.outputdir='/projects/stan/biomakers/IL6_reho'
        ffp.make_files_and_run()
        """

    def __init__(
        self,
        columns,
        data,
        basename="design",
        outputdir="./",
        gmv=False,
        demean=False,
        round_to=3,
        ffill=False,
        image_col="",
        mask_image=None,
        zthresh=2.33,
        pthresh=0.05,
        two_sided=True,
        bg_image=None,
        results_dir="results",
    ):
        # Check variable type for required arguments
        if not isinstance(columns, list):
            raise Exception("'columns' must be a list")
        if not isinstance(data, pd.DataFrame):
            raise Exception("'data' must be a Pandas DataFrame")

        # Check that FSL is installed and configured.
        FSLDIR = os.getenv("FSLDIR")
        if FSLDIR is None:
            raise Exception(
                """FSLDIR environment variable not set.
                Is FSL installed and configured correctly?"""
            )
        if not os.path.exists(FSLDIR):
            raise Exception(
                """FSLDIR not found at: {}.
                Is FSL installed and configured correctly?""".format(
                    self.fsldir
                )
            )

        self.columns = columns
        self.dataframe = data
        self.basename = basename
        self.outputdir = os.path.abspath(os.path.expanduser(outputdir))
        self.gmv = gmv
        self.demean_matcols = demean
        self.decimal_places = round_to
        self.ffill = ffill
        self.mat = None
        self.con = None
        self.grp = None
        self.filt_func_data = "Set this to a path and link in outputdir?"
        self.mask_image = mask_image
        self.image_col = image_col
        self.fsldir = FSLDIR
        self.default_mask = os.path.join(
            self.fsldir, "data/standard/MNI152_T1_2mm_brain_mask.nii.gz"
        )
        self.zthresh = zthresh
        self.pthresh = pthresh
        self.two_sided = two_sided
        self.results_dir = results_dir
        if bg_image is None:
            self.bg_image = os.path.join(
                self.fsldir, "data/standard/MNI152_T1_2mm_brain.nii.gz"
            )
        else:
            self.bg_image = bg_image

    def make_filtered_func(self):
        """
        Merge images specified in 'image_col' to 4D filtered_func_data. save in 'outputdir'
        These images will be aligned with the values of interest specified in 'columns'
        """
        if self.image_col == "":
            print(
                """'image_col' variable must be set to the name of the column
            containing the full paths to images of interest before
            a filtered_func_data 4D file can be created."""
            )
        try:
            cmdstr = "fslmerge -t {}/filtered_func_data ".format(
                self.outputdir)
            for s in self.dataframe[self.image_col]:
                cmdstr += s + " "
        except:
            raise Exception(
                """Not able to merge strings from {}
                to {}/filtered_func_data""".format(
                    self.image_col, self.outputdir
                )
            )

        # create/check outputdir if no exception thrown
        self.mk_outputdir()

        print("""Merging images to:
            {}/filtered_func_data""".format(self.outputdir))

        x = os.system(cmdstr)

        if x != 0:
            raise Exception(
                """An error ({}) occurred while attempting to merge images
                from dataframe column '{}'
                to: {}/filtered_func_data""".format(
                    x, self.image_col, self.outputdir
                )
            )
        else:
            return 0

    def mk_summary(self):
        """
        Return formatted string representation of
        the current parameters that will be used
        to create feat files and run analyses.
        """

        self.selfupdate()

        if self.mat is not None:
            mat_hdr = "\n\t\t  ".join(
                self.mat.split("\n")[:5]) + "\n\t\t  ...etc"
        else:
            mat_hdr = "Not Created Yet."

        if self.grp is not None:
            grp_hdr = "\n\t\t  ".join(
                self.grp.split("\n")[:5]) + "\n\t\t  ...etc"
        else:
            grp_hdr = "Not Created Yet."

        if self.con is not None:
            n_cols = len(self.columns) + 4
            con_hdr = "\n\t\t  ".join(self.con.split(
                "\n")[:n_cols]) + "\n\t\t  ...etc"
        else:
            con_hdr = "Not Created Yet."

        # Hate to hard-code these things.
        # Maybe come back to this later and update
        cols = "columns         : {}\n".format(self.columns)
        dat = "data            : {}\n".format(self.dataframe.columns)
        base = "basename        : {}\n".format(self.basename)
        out = "outputdir       : {}\n".format(self.outputdir)
        gv = "gmv             : {}\n".format(self.gmv)
        dm = "demean_matcols  : {}\n".format(self.demean_matcols)
        dec = "decimal_places  : {}\n".format(self.decimal_places)
        fill = "ffill           : {}\n".format(self.ffill)
        mat = "mat             : {}\n".format(mat_hdr)
        con = "con             : {}\n".format(con_hdr)
        grp = "grp             : {}\n".format(grp_hdr)
        ffd = "filt_func_data  : {}\n".format(self.filt_func_data)
        fsldir = "FSLDIR          : {}\n".format(self.fsldir)
        mask = "mask_image      : {}\n".format(self.mask_image)
        pval = "pthresh         : {}\n".format(self.pthresh)
        zstat = "zthresh         : {}\n".format(self.zthresh)
        twosided = "two_sided       : {}\n".format(self.two_sided)

        outstr = "Object Values:\n"

        outstr += cols + dat + base + out + gv + dm + dec
        outstr += fill + mat + con + grp + ffd + fsldir
        outstr += mask + pval + zstat + twosided

        return outstr

    def show_parameters(self):
        """
        Print summary of current object parameters.
        """
        print(self.mk_summary())

    def save_paramaters(self):
        """
        Write summary of current object parameters to 'outputdir'.
        Will create outputdir if it does not exist.
        """
        self.mk_outputdir()  # check that we have a directory to write in
        with open(os.path.join(self.outputdir, "featfile_params.txt"), "w") as f:
            f.write(self.mk_summary())

    def write_feats(self):
        """
        Write feat mat, con and grp files to 'outputdir'
        Will create outputdir if it does not exist.
        """

        self.mk_outputdir()
        self.selfupdate()

        with open(os.path.join(self.outputdir, "{}.mat".format(self.basename)), "w") as f:
            f.write(self.mat)

        with open(os.path.join(self.outputdir, "{}.con".format(self.basename)), "w") as f:
            f.write(self.con)

        with open(os.path.join(self.outputdir, "{}.grp".format(self.basename)), "w") as f:
            f.write(self.grp)

    def mk_designcon(self):
        """Create design.mat string representation and save in object variable."""
        self.con = ""
        con_tmplt = "ContrastName{}\t{}"

        if self.gmv is True:
            numcols = len(self.columns) + 1
        else:
            numcols = len(self.columns)

        for k in range(len(self.columns)):
            self.con += con_tmplt.format(k + 1, self.columns[k]) + "\n"

        if self.gmv is True:
            self.con += con_tmplt.format(k + 2, "grand_mean\n")

        self.con += "/NumWaves\t{}\n".format(numcols)
        self.con += "/NumContrasts\t{}\n".format(numcols)
        self.con += "/Matrix\n"

        I = np.identity(numcols, dtype=int)

        for k in range(len(I)):
            s = I[k].astype(str)
            self.con += "\t".join(s) + "\n"

    def mk_outputdir(self):
        """
        Checks for existence of 'outputdir', and creates folders if they do not already exist.
        """
        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

    def mk_designgrp(self):
        """Create design.grp string representation and save in object variable"""

        grp_tmplt = "/NumWaves {0}\n/NumPoints {1}\n/Matrix\n"
        self.grp = grp_tmplt.format(1, len(self.dataframe))
        I = np.ones((len(self.dataframe)), dtype=int)

        for s in I:
            self.grp += "{}\n".format(s)

    def mk_designmat(self):
        """Create design.mat string representation and save in object variable"""

        self.mat = ""
        mat_tmplt = "/NumWaves {0}\n/NumPoints {1}\n/Matrix\n"
        X = self.dataframe[self.columns].copy()

        if self.demean_matcols is True:
            for c in X.columns:
                X[c] = X[c] - X[c].mean()

        X = X.round(self.decimal_places)

        if self.gmv is True:
            X["gmv"] = 1

        self.mat = mat_tmplt.format(len(X.columns), len(X))

        if self.ffill is True:
            X.ffill(inplace=True)

        for k in range(len(X)):
            vals = X.iloc[k].astype(str)
            self.mat += "\t".join(vals) + "\n"

    def cp_mask_file(self):
        """
        Copy mask file to 'outputdir'. If no mask is specified, the default
        2mm brain mask image provided with FSL will be used.
        """
        if self.mask_image is None:
            maskstr = os.path.join(self.fsldir, self.default_mask)
            print("Warning: copying default 2mm mask image from {}".format(maskstr))
        else:
            maskstr = self.mask_image
        cmd = "cp {} {}/{}".format(maskstr, self.outputdir, "mask.nii.gz ")
        x = os.system(cmd)
        return x

    def selfupdate(self):
        """
        Update string representations for mat, con, and grp files, using all current parameters,
        and store in class object.
        """
        self.mk_designcon()
        self.mk_designmat()
        self.mk_designgrp()

    def run_feat_files(self):
        """
        Run flameo on feat files and filtered_func_data in 'outputdir'
        These files must already exist in 'outputdir'
        """
        flameo = "flameo --cope={0} --mask={1} --dm={2}.mat --tc={2}.con --cs={2}.grp --logdir={3} --runmode=ols"
        mask = os.path.join(self.outputdir, "mask")
        cope = os.path.join(self.outputdir, "filtered_func_data")
        filebase = os.path.join(self.outputdir, self.basename)
        logdir = os.path.join(self.outputdir, self.results_dir)

        if os.path.exists(logdir) is True:
            #             os.rename(logdir, logdir+'.old')
            os.rmdir(resultsdir)

        cmdstr = flameo.format(cope, mask, filebase, logdir)
        print(cmdstr)
        x = os.system(cmdstr)
        return x

    def make_files_and_run(self):
        """
        Create feat files and filtered_func_data, then run flameo in 'outputdir'
        """
        self.write_feats()
        self.mk_outputdir()
        self.make_filtered_func()
        self.cp_mask_file()
        self.run_feat_files()
        self.thresh_zstats()
        self.webpage_for_thresh_zstats()

    # Usage: easythresh <raw_zstat> <brain_mask> <cluster_z_thresh> <cluster_prob_thresh> <background_image> <output_root> [--mm]
    def thresh_zstats(self):
        """
        Apply easythresh to zstats in results dir. 
        """

        tmplt = "cd {} ; easythresh {} {} {} {} {} {} --mm"
        mask = os.path.join(self.outputdir, "mask")
        resultsdir = os.path.join(self.outputdir, "results")

        for k in range(
            len(self.columns)
        ):  # using len(self.columns) will prevent GMV from rendering
            zstat = "zstat{}".format(k + 1)
            cmdstr = tmplt.format(
                resultsdir,
                zstat,
                mask,
                self.zthresh,
                self.pthresh,
                self.bg_image,
                "pos_{}_zstat{}".format(self.columns[k], k + 1),
            )
            print(cmdstr)
            os.system(cmdstr)
            if self.two_sided is True:
                invstat = "neg_zstat{}".format(k + 1)
                cmdstr = "cd {0} ; fslmaths {1} -mul -1 {2}".format(
                    resultsdir, zstat, invstat
                )
                os.system(cmdstr)
                print(cmdstr)
                cmdstr = tmplt.format(
                    resultsdir,
                    invstat,
                    mask,
                    self.zthresh,
                    self.pthresh,
                    self.bg_image,
                    "neg_{}_zstat{}".format(self.columns[k], k + 1),
                )
                print(cmdstr)
                os.system(cmdstr)

    def webpage_for_thresh_zstats(self):
        import glob
        from dominate import document
        from dominate.tags import h1, img, div, p, b

        imgs = glob.glob("{}/results/*.png".format(self.outputdir))

        page_title = "Results: "
        for c in self.columns:
            page_title += c + ", "

        with document(title=page_title) as doc:
            h1(page_title)
            for path in imgs:
                txt = path.split("/")[-1]
                div(p(b(txt)))
                div(img(src=path), _class="photo")

        with open("{}/results_rendered.html".format(self.outputdir), "w") as f:
            f.write(doc.render())


# TODO: adapt for use as a command-line script
#      Will need to import argparse, and load
#      pandas dataframe from string argument.
#      Probably additional logic to define
#      how far to process (just feat files or
#      run the full pipeline, etc.?)
# def main():
#     #If called from script, run with provided variables
#     self.make_files_and_run_from_scratch()

# if __name__ == "__main__":
#     #if called as a script
#     main()
