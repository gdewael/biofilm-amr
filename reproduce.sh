python scripts/model_variants.py data/evolved/MIC_WGS.csv --save_weights_path_prefix ./res/MIC_WGS_ --save_preds_path_prefix ./res/MIC_WGS_ --save_figs_path_prefix ./res/MIC_WGS_
python scripts/model_variants.py data/evolved/BPC_WGS.csv --save_weights_path_prefix ./res/BPC_WGS_ --save_preds_path_prefix ./res/BPC_WGS_ --save_figs_path_prefix ./res/BPC_WGS_

python scripts/model_maldi.py data/evolved/MIC_MALDI.csv --save_weights_path_prefix ./res/MIC_MALDI_ --save_preds_path_prefix ./res/MIC_MALDI_ --save_figs_path_prefix ./res/MIC_MALDI_
python scripts/model_maldi.py data/evolved/BPC_MALDI.csv --save_weights_path_prefix ./res/BPC_MALDI_ --save_preds_path_prefix ./res/BPC_MALDI_ --save_figs_path_prefix ./res/BPC_MALDI_

python scripts/model_imc.py data/evolved/MIC_IMC.csv --save_weights_path_prefix ./res/MIC_IMC_ --save_preds_path_prefix ./res/MIC_IMC_ --save_figs_path_prefix ./res/MIC_IMC_
python scripts/model_imc.py data/evolved/BPC_IMC.csv --save_weights_path_prefix ./res/BPC_IMC_ --save_preds_path_prefix ./res/BPC_IMC_ --save_figs_path_prefix ./res/BPC_IMC_

python scripts/model_raman.py data/evolved/MIC_RAMAN_532.csv --save_weights_path_prefix ./res/MIC_RAMAN532_ --save_preds_path_prefix ./res/MIC_RAMAN532_ --save_figs_path_prefix ./res/MIC_RAMAN532_
python scripts/model_raman.py data/evolved/BPC_RAMAN_532.csv --save_weights_path_prefix ./res/BPC_RAMAN532_ --save_preds_path_prefix ./res/BPC_RAMAN532_ --save_figs_path_prefix ./res/BPC_RAMAN532_
python scripts/model_raman.py data/evolved/MIC_RAMAN_785.csv --save_weights_path_prefix ./res/MIC_RAMAN785_ --save_preds_path_prefix ./res/MIC_RAMAN785_ --save_figs_path_prefix ./res/MIC_RAMAN785_
python scripts/model_raman.py data/evolved/BPC_RAMAN_785.csv --save_weights_path_prefix ./res/BPC_RAMAN785_ --save_preds_path_prefix ./res/BPC_RAMAN785_ --save_figs_path_prefix ./res/BPC_RAMAN785_
python scripts/model_raman.py data/evolved/MIC_RAMAN_MX.csv --save_weights_path_prefix ./res/MIC_RAMANMX_ --save_preds_path_prefix ./res/MIC_RAMANMX_ --save_figs_path_prefix ./res/MIC_RAMANMX_
python scripts/model_raman.py data/evolved/BPC_RAMAN_MX.csv --save_weights_path_prefix ./res/BPC_RAMANMX_ --save_preds_path_prefix ./res/BPC_RAMANMX_ --save_figs_path_prefix ./res/BPC_RAMANMX_

python scripts/model_stacking.py ./res/MIC_WGS_predictions.csv ./res/MIC_MALDI_predictions.csv ./res/MIC_RAMANMX_predictions.csv ./res/MIC_IMC_predictions.csv MIC --save_weights_path_prefix ./res/MIC_STACKING_ --save_preds_path_prefix ./res/MIC_STACKING_ --save_figs_path_prefix ./res/MIC_STACKING_
python scripts/model_stacking.py ./res/BPC_WGS_predictions.csv ./res/BPC_MALDI_predictions.csv ./res/BPC_RAMANMX_predictions.csv ./res/BPC_IMC_predictions.csv BPC --save_weights_path_prefix ./res/BPC_STACKING_ --save_preds_path_prefix ./res/BPC_STACKING_ --save_figs_path_prefix ./res/BPC_STACKING_

python scripts/model_maldi_isolates.py data/evolved/MIC_MALDI.csv data/isolates/MIC_MALDI.csv --save_preds_path_prefix ./res/ISOLATES_MIC_MALDI_
python scripts/model_maldi_isolates.py data/evolved/BPC_MALDI.csv data/isolates/BPC_MALDI.csv --save_preds_path_prefix ./res/ISOLATES_BPC_MALDI_

python scripts/model_imc_isolates.py data/evolved/MIC_IMC.csv data/isolates/MIC_IMC.csv --save_preds_path_prefix ./res/ISOLATES_MIC_IMC_
python scripts/model_imc_isolates.py data/evolved/BPC_IMC.csv data/isolates/BPC_IMC.csv --save_preds_path_prefix ./res/ISOLATES_BPC_IMC_

python scripts/model_raman_isolates.py data/evolved/MIC_RAMAN_532.csv data/isolates/MIC_RAMAN_532.csv --save_preds_path_prefix ./res/ISOLATES_MIC_RAMAN532_
python scripts/model_raman_isolates.py data/evolved/BPC_RAMAN_532.csv data/isolates/BPC_RAMAN_532.csv --save_preds_path_prefix ./res/ISOLATES_BPC_RAMAN532_
python scripts/model_raman_isolates.py data/evolved/MIC_RAMAN_785.csv data/isolates/MIC_RAMAN_785.csv --save_preds_path_prefix ./res/ISOLATES_MIC_RAMAN785_
python scripts/model_raman_isolates.py data/evolved/BPC_RAMAN_785.csv data/isolates/BPC_RAMAN_785.csv --save_preds_path_prefix ./res/ISOLATES_BPC_RAMAN785_
python scripts/model_raman_isolates.py data/evolved/MIC_RAMAN_MX.csv data/isolates/MIC_RAMAN_MX.csv --save_preds_path_prefix ./res/ISOLATES_MIC_RAMANMX_
python scripts/model_raman_isolates.py data/evolved/BPC_RAMAN_MX.csv data/isolates/BPC_RAMAN_MX.csv --save_preds_path_prefix ./res/ISOLATES_BPC_RAMANMX_