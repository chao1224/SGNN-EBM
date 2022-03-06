rm -rf output/mtl
rm -rf output/uw
rm -rf output/gradnorm
rm -rf output/dwa
rm -rf output/lbtw

rm -rf model_weight/mtl
rm -rf model_weight/uw
rm -rf model_weight/gradnorm
rm -rf model_weight/dwa
rm -rf model_weight/lbtw

ls output
ls model_weight

bash submit_baseline_mtl.sh
bash submit_baseline_uw.sh
bash submit_baseline_gradnorm.sh
bash submit_baseline_dwa.sh
bash submit_baseline_lbtw.sh

