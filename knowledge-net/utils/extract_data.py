
"""Extract Features Data

This script extracts the data belonging to features listed in a 
features file from a pre-formatted data file.
"""



import pandas as pd
import argparse



def parse_args():
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
            "data_file",
            help="""Specify the path to the data file from which to extract 
                    data.""") 
    parser.add_argument(
            "feature_file",
            help="""Specify the path to the file listing the features to 
                    extract from the main data file.
                    If no outfile path is provided using the -o 
                    option, the outfile will be written into this same 
                    directory.""")
    parser.add_argument(
            "-t", "--target",
            help="""Specify whether to specify just a single target, reducing 
                    the problem to a binary classification.""")

    parser.add_argument(
            "-o", "--outfile",
            help="""Specify the path to the file to be written.""")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    # Parse command line arguments
    args = parse_args()
    dfile = args.data_file
    efile = args.feature_file
    target = args.target
    outfile = args.outfile
    
    # Set the path to the outfile if none is provided.
    if not outfile:
        outfile = efile.rsplit("/", 1)[0] + "/data.tsv"

    # Load in the data file and the extraction file.
    data_df = pd.read_csv(dfile, sep="\t")
    extract_list = list(pd.read_csv(efile, sep="\t").iloc[:, 1])
    
    # Change the non-target outputs to "Other" (if -t arg is used).
    if target != None:
        data_df["Output"][data_df["Output"] != target] = "Other"

    # Extract the data and write to file.
    extract_list.append("Output")
    out_df = data_df[extract_list]
    out_df.to_csv(outfile, sep="\t", index=False)

