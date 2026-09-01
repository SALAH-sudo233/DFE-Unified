#!/bin/bash
# Multi-pocket eval for DF 500K (30 pockets) - Pure DF model
set -e

cd /workspace/ayb/Pocket2Mol
export PYTHONPATH=/workspace/ayb/Pocket2Mol:$PYTHONPATH
PY=/workspace/ayb/miniconda3/envs/zatom310/bin/python
EVAL_DOCK=/workspace/ayb/Pocket2Mol/evaluate_docking_fixed.py
PDBBIND=/workspace/ayb/data/pdbbind/v2020
GPU=1

POCKETS="10gs 1a1e 1a30 1a4k 1a94 1a9q 2yi0 4afg 4kzu 4zl4 1ezq 1f73 1fm9 1g30 1sgu 1ui0 1wur 1zgi 2iko 3ddg 3ebp 3fv3 3i4b 3ove 4qxo 5jxq 5u14 6fag 6guc 6qr1"

declare -A CENTER
CENTER[10gs]="9.877,6.630,27.936"
CENTER[1a1e]="41.769,-7.402,42.155"
CENTER[1a30]="8.446,25.365,4.394"
CENTER[1a4k]="21.494,9.633,1.321"
CENTER[1a94]="58.464,17.083,32.947"
CENTER[1a9q]="24.689,91.079,73.618"
CENTER[2yi0]="32.202,8.960,23.965"
CENTER[4afg]="45.409,-1.919,14.951"
CENTER[4kzu]="-6.608,37.029,-12.284"
CENTER[4zl4]="-2.892,98.626,41.948"
CENTER[1ezq]="7.329,4.831,22.148"
CENTER[1f73]="-4.960,44.701,51.199"
CENTER[1fm9]="11.930,-6.674,8.931"
CENTER[1g30]="17.611,-14.911,23.089"
CENTER[1sgu]="18.124,0.198,10.715"
CENTER[1ui0]="22.748,9.066,11.507"
CENTER[1wur]="-26.818,-18.633,5.623"
CENTER[1zgi]="16.939,-14.146,23.071"
CENTER[2iko]="-12.793,25.151,-36.155"
CENTER[3ddg]="30.232,66.607,10.333"
CENTER[3ebp]="39.949,36.765,29.587"
CENTER[3fv3]="19.786,24.220,59.495"
CENTER[3i4b]="37.748,35.383,55.993"
CENTER[3ove]="-8.229,8.246,-0.829"
CENTER[4qxo]="-20.471,52.110,-2.772"
CENTER[5jxq]="107.020,19.947,15.643"
CENTER[5u14]="70.750,-0.166,101.644"
CENTER[6fag]="12.266,35.006,16.656"
CENTER[6guc]="-7.658,-22.210,22.679"
CENTER[6qr1]="-21.760,20.542,-10.639"

BBOX=23.0
NUM_SAMPLES=100

MODEL_NAME="df500k"
CKPT="./logs/checkpoints/500000.pt"
CONFIG="configs/sample_df_500k.yml"

RESULTS_FILE=outputs/multi_pocket_eval_df500k_30pocket_results.json
echo "{}" > $RESULTS_FILE

echo "=== DF 500K 30-Pocket Eval Started $(date) ==="
echo "Pockets: $POCKETS"
echo "Num samples: $NUM_SAMPLES"
echo "GPU: $GPU"
echo "Checkpoint: $CKPT"
echo ""

# Create modified config with 100 samples
MOD_CONFIG="/tmp/df500k_100samples.yml"
$PY -c "
import yaml
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
c['sample']['num_samples'] = $NUM_SAMPLES
c['sample']['beam_size'] = 300
with open('$MOD_CONFIG', 'w') as f:
    yaml.dump(c, f)
"

for POCKET in $POCKETS; do
    echo ""
    echo "--- $MODEL_NAME x $POCKET ---"
    echo "Time: $(date)"
    
    PROTEIN_PDB="$PDBBIND/$POCKET/${POCKET}_protein.pdb"
    POCKET_PDB="$PDBBIND/$POCKET/${POCKET}_prot/${POCKET}_p_pocket_10.0.pdb"
    
    if [ ! -f "$PROTEIN_PDB" ]; then
        echo "ERROR: Protein PDB not found: $PROTEIN_PDB"
        continue
    fi
    if [ ! -f "$POCKET_PDB" ]; then
        POCKET_PDB="$PDBBIND/$POCKET/${POCKET}_pocket.pdb"
    fi
    if [ ! -f "$POCKET_PDB" ]; then
        echo "ERROR: Pocket PDB not found for $POCKET"
        continue
    fi
    
    CENTER_VAL=${CENTER[$POCKET]}
    OUTDIR="outputs/eval_${MODEL_NAME}_30pocket_${POCKET}"
    rm -rf "$OUTDIR" 2>/dev/null
    mkdir -p "$OUTDIR"
    
    echo "Step 1: Sampling $NUM_SAMPLES molecules..."
    SAMPLE_START=$(date +%s)
    
    CUDA_VISIBLE_DEVICES=$GPU $PY -u sample_for_pdb_nodisk.py \
        --pdb_path "$PROTEIN_PDB" \
        --center "$CENTER_VAL" \
        --bbox_size $BBOX \
        --config "$MOD_CONFIG" \
        --device cuda \
        --outdir "outputs/" 2>&1 | tail -30
    
    SAMPLE_END=$(date +%s)
    SAMPLE_TIME=$((SAMPLE_END - SAMPLE_START))
    echo "Sampling took ${SAMPLE_TIME}s"
    
    PDB_BASE=$(basename "$PROTEIN_PDB" .pdb)
    CONFIG_BASE=$(basename "$CONFIG" .yml)
    SAMPLE_DIR=$(ls -dt outputs/${CONFIG_BASE}_${PDB_BASE}* 2>/dev/null | head -1)
    
    if [ -z "$SAMPLE_DIR" ]; then
        SAMPLE_DIR=$(ls -dt outputs/*${PDB_BASE}* 2>/dev/null | head -1)
    fi
    
    if [ -z "$SAMPLE_DIR" ]; then
        echo "ERROR: No sample directory found"
        continue
    fi
    
    SDF_DIR="$SAMPLE_DIR/SDF"
    if [ ! -d "$SDF_DIR" ]; then
        echo "ERROR: No SDF directory in $SAMPLE_DIR"
        rm -rf "$SAMPLE_DIR"
        continue
    fi
    
    NUM_SDF=$(ls "$SDF_DIR"/*.sdf 2>/dev/null | wc -l)
    echo "Generated $NUM_SDF molecules"
    
    if [ "$NUM_SDF" -eq 0 ]; then
        echo "ERROR: No molecules generated"
        rm -rf "$SAMPLE_DIR"
        continue
    fi
    
    echo "Step 2: Merging SDFs..."
    MERGED_SDF="$OUTDIR/merged_all.sdf"
    $PY convert_samples_to_sdf.py --input "$SAMPLE_DIR" --output "$MERGED_SDF" 2>&1 | tail -3
    
    if [ ! -f "$MERGED_SDF" ]; then
        cat "$SDF_DIR"/*.sdf > "$MERGED_SDF"
    fi
    
    [ -f "$SAMPLE_DIR/SMILES.txt" ] && cp "$SAMPLE_DIR/SMILES.txt" "$OUTDIR/SMILES.txt"
    rm -rf "$SAMPLE_DIR"
    
    echo "Step 3: Docking evaluation..."
    DOCK_START=$(date +%s)
    DOCK_OUTPUT="$OUTDIR/docking_results.json"
    
    CUDA_VISIBLE_DEVICES=$GPU $PY -u $EVAL_DOCK \
        --sdf "$MERGED_SDF" \
        --pocket_pdb "$POCKET_PDB" \
        --output "$DOCK_OUTPUT" \
        --exhaustiveness 8 \
        --num_modes 9 2>&1 | tail -20
    
    DOCK_END=$(date +%s)
    DOCK_TIME=$((DOCK_END - DOCK_START))
    echo "Docking took ${DOCK_TIME}s"
    
    echo "Step 4: Computing metrics..."
    $PY /workspace/ayb/Pocket2Mol/compute_metrics.py "$DOCK_OUTPUT" "$RESULTS_FILE" "$POCKET" 2>&1 | tail -5
    
    echo "--- Done $POCKET ---"
done

echo ""
echo "=== All 30 pockets completed $(date) ==="
$PY -c "
import json
with open('$RESULTS_FILE') as f:
    results = json.load(f)
print(f'Total pockets evaluated: {len(results)}')
for pocket, m in results.items():
    ds = m.get('docking_score_median', 'N/A')
    print(f'  {pocket}: valid={m[\"valid_molecules\"]}, dock_median={ds}')
"
