# # mjx_dmc
# sbatch scripts/killarney_july/slurm_dmc.sh mjx_dmc unused mjx_dmc_small_data small
# sbatch scripts/killarney_july/slurm_dmc.sh mjx_dmc unused mjx_dmc_medium_data medium
# sbatch scripts/killarney_july/slurm_dmc.sh mjx_dmc unused mjx_dmc_large_data large
# 
# sbatch scripts/killarney_july/slurm_dmc_aux_ablation.sh mjx_dmc unused mjx_dmc_small_data small
# sbatch scripts/killarney_july/slurm_dmc_aux_ablation.sh mjx_dmc unused mjx_dmc_medium_data medium
# sbatch scripts/killarney_july/slurm_dmc_aux_ablation.sh mjx_dmc unused mjx_dmc_large_data large
# 
# sbatch scripts/killarney_july/slurm_dmc_gauss_ablation.sh mjx_dmc unused mjx_dmc_small_data small
# sbatch scripts/killarney_july/slurm_dmc_gauss_ablation.sh mjx_dmc unused mjx_dmc_medium_data medium
# sbatch scripts/killarney_july/slurm_dmc_gauss_ablation.sh mjx_dmc unused mjx_dmc_large_data large
# 
# sbatch scripts/killarney_july/slurm_dmc_norm_ablation.sh mjx_dmc unused mjx_dmc_small_data small
# sbatch scripts/killarney_july/slurm_dmc_norm_ablation.sh mjx_dmc unused mjx_dmc_medium_data medium
# sbatch scripts/killarney_july/slurm_dmc_norm_ablation.sh mjx_dmc unused mjx_dmc_large_data large
# 
# sbatch scripts/killarney_july/slurm_dmc_op_clip_ablation.sh mjx_dmc unused mjx_dmc_small_data full small
# sbatch scripts/killarney_july/slurm_dmc_op_clip_ablation.sh mjx_dmc unused mjx_dmc_small_data value small
# 
# sbatch scripts/killarney_july/slurm_dmc_op_clip_ablation.sh mjx_dmc unused mjx_dmc_medium_data full medium
# sbatch scripts/killarney_july/slurm_dmc_op_clip_ablation.sh mjx_dmc unused mjx_dmc_medium_data value medium
# 
# sbatch scripts/killarney_july/slurm_dmc_op_clip_ablation.sh mjx_dmc unused mjx_dmc_large_data full large
# sbatch scripts/killarney_july/slurm_dmc_op_clip_ablation.sh mjx_dmc unused mjx_dmc_large_data value large

# mjx_humanoid
sbatch scripts/killarney_july/slurm_hum.sh mjx_humanoid unused mjx_humanoid_small_data asymmetric_set,small
sbatch scripts/killarney_july/slurm_hum.sh mjx_humanoid unused mjx_humanoid_large_data asymmetric_set,large

sbatch scripts/killarney_july/slurm_hum_aux_ablation.sh mjx_humanoid unused mjx_humanoid_small_data asymmetric_set,small
sbatch scripts/killarney_july/slurm_hum_aux_ablation.sh mjx_humanoid unused mjx_humanoid_large_data asymmetric_set,large

sbatch scripts/killarney_july/slurm_hum_gauss_ablation.sh mjx_humanoid unused mjx_humanoid_small_data asymmetric_set,small
sbatch scripts/killarney_july/slurm_hum_gauss_ablation.sh mjx_humanoid unused mjx_humanoid_large_data asymmetric_set,large

sbatch scripts/killarney_july/slurm_hum_norm_ablation.sh mjx_humanoid unused mjx_humanoid_small_data asymmetric_set,small
sbatch scripts/killarney_july/slurm_hum_norm_ablation.sh mjx_humanoid unused mjx_humanoid_large_data asymmetric_set,large

sbatch scripts/killarney_july/slurm_hum_op_clip_ablation.sh mjx_humanoid unused mjx_humanoid_small_data full asymmetric_set,small
sbatch scripts/killarney_july/slurm_hum_op_clip_ablation.sh mjx_humanoid unused mjx_humanoid_small_data value asymmetric_set,small

sbatch scripts/killarney_july/slurm_hum_op_clip_ablation.sh mjx_humanoid unused mjx_humanoid_medium_data full asymmetric_set,medium
sbatch scripts/killarney_july/slurm_hum_op_clip_ablation.sh mjx_humanoid unused mjx_humanoid_medium_data value asymmetric_set,medium

sbatch scripts/killarney_july/slurm_hum_op_clip_ablation.sh mjx_humanoid unused mjx_humanoid_large_data full asymmetric_set,large
sbatch scripts/killarney_july/slurm_hum_op_clip_ablation.sh mjx_humanoid unused mjx_humanoid_large_data value asymmetric_set,large
