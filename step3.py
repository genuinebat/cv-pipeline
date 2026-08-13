import os
from label import LabelPreperation 

if __name__ == "__main__":
    CONV_INPUT = "./4validated_imgs"
    CONV_OUTPUT = "./5ready_imgs"

    SPLIT_TRAIN = "./dataset/train"
    SPLIT_VAL = "./dataset/val" 

    lp = LabelPreperation()
    lp.convert_validated_json_labels_to_text(CONV_INPUT, CONV_OUTPUT)

    labels = [1 if f.endswith(".txt") else 0 for f in os.list_dir(OUTPUT)]
    n = sum(labels) 

    t, v, te = lp.split_data(input_dir = CONV_OUTPUT, output_train=SPLIT_TRAIN, output_val=SPLIT_VAL)

    print(f"""
    ------------ Phase 3 Results ------------
    {n} label{"" if n == 1 else "s"} converted to text files

    Added {n} total label and image pair{"" if n == 1 else "s"} to dataset
    Added {t} label and image pair{"" if n == 1 else "s"} to training dataset
    Added {n} label and image pair{"" if n == 1 else "s"} to validation dataset
    Added {te} label and image pair{"" if n == 1 else "s"} to testing dataset
    """)