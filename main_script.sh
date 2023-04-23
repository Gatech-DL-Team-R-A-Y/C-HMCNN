for dataset in "enron_others" "diatoms_others" "imclef07a_others" "imclef07d_others" "cellcycle_FUN" "derisi_FUN" "eisen_FUN" "expr_FUN" "gasch1_FUN" "gasch2_FUN" "seq_FUN" "spo_FUN" "cellcycle_GO" "derisi_GO" "eisen_GO" "expr_GO" "gasch1_GO" "gasch2_GO" "seq_GO" "spo_GO"
do
    python main.py --dataset $dataset --seed 0
done


