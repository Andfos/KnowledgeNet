
"""Generate Ontology File

This script will use the modules.csv file and the features.txt file to generate 
the ontology.txt file showing the relationships between inputs and modules.
"""

import sys
import argparse
import pandas as pd




def parse_args():
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
            "module_file",
            help="""Specify the path to the modules.csv file that groups
                    features.""")

    parser.add_argument(
            "feature_file",
            help="""Specify the path to the features file containing the 
                    list of all features to be included in the ontology.""")
    
    parser.add_argument(
            "-u", "--unconstrain",
            help="""Specify whether to ignore the modules.csv file and feed all 
                    features into the same first module.""", 
            action="store_true",
            default=False)
    
    parser.add_argument(
            "-n", "--num_hidden",
            help="Specify the number of hidden layers (default = 2).", 
            default=2)

    parser.add_argument(
            "-p", "--prefix",
            help="""Specify the prefix of the hidden layers (default = L). The 
                    suffix will be the number of the hidden layer, beginning 
                    with 2.""",
            default="L")

    args = parser.parse_args()
    return args



if __name__ == "__main__":

    # Parse the command line arguments.
    args = parse_args()
    module_file = args.module_file
    feature_file = args.feature_file
    unconstrain = args.unconstrain
    num_hidden_layers = int(args.num_hidden)
    prefix = args.prefix

    # Load in the modules.csv file and the features.tsv file.
    out_file = feature_file.rsplit("/", 1)[0] + "/ontology.txt"
    modules_df = pd.read_csv(module_file)
    features_df = pd.read_csv(feature_file, sep="\t")

    # Instantiate lists to record inputs, modules, and relations.
    inputs = []
    modules = []
    relations = []

    # Iterate over every 
    
    features = list(features_df["Feature"])
    all_features = list(modules_df["Feature"])
    all_features = [f.lower() for f in all_features]
    all_modules = list(modules_df["Module"])

    for inp in features:
        ind = features.index(inp)
        mod = all_modules[ind]
        if unconstrain:
            modules.append("L1")
        else:
            modules.append(mod)
        
        inputs.append(inp)
        relations.append("gene")
        
        # Break the code if any inputs or modules have disallowed characters.
        disallowed = [":", "/", "\\", ";", "_"]
        if any((char in mod) or (char in inp) for char in disallowed):    
            print(f"Characters {', '.join(disallowed)} not allowed in names.")
            sys.exit(1)

    # Add the relationships between lower-level modules and higher-level modules.
    mod_set = set(modules)
    
    

    for l in range(2, num_hidden_layers + 1):

        # Add a relationship between each of the unique modules and the first
        # non-specific layer.
        if l == 2:
            modules = [f"{prefix}{str(l)}"] * len(mod_set) + modules
            inputs = list(mod_set) + inputs
            relations = ["default"] * len(mod_set) + relations
        else:
            modules = [f"{prefix}{str(l)}"] + modules
            inputs = [f"{prefix}{str(l-1)}"] + inputs
            relations = ["default"] + relations
            
    # Add relationship between the last hidden layer and the prediction layer.
    modules = ["Prediction"] + modules
    inputs = [f"{prefix}{str(num_hidden_layers)}"] + inputs
    relations = ["default"] + relations

    # Place the lists in a data frame and write the output to ontology.txt.
    output_df = pd.DataFrame()
    output_df["Module"] = modules
    output_df["Input"] = inputs
    output_df["Relation"] = relations
    output_df.to_csv(out_file, sep="\t", index=False, header=False)
