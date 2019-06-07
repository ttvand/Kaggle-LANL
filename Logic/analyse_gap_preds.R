library(data.table)

analysis_file <- c("train", "valid", "test")[2]
confident_limit <- 0.999

data_folder <- "/media/tom/cbd_drive/Kaggle/LANL/Data/"
data <- fread(file.path(data_folder, paste0(
  "gap_model_aligned_predictions_", analysis_file, ".csv")))

num_files <- ncol(data)
num_obs <- nrow(data)
predictions <- as.matrix(data)
confident_gap_ids <- which(predictions > confident_limit)
confident_cols <- ceiling((confident_gap_ids-0.5)/num_obs)
confident_rows <- confident_gap_ids - (confident_cols-1)*num_obs

all_confident_mods <- vector(mode="character", length=num_files)
all_consensus_files <- vector(mode="character", length=4096)
consensus_files_count <- vector(mode="numeric", length=4096)
for(file_id in 1:num_files){
  confident_ids <- which(confident_cols == file_id)
  mod_ids <- confident_rows[confident_ids] %% 4096
  cat("File:", file_id, mod_ids, "\n")
  if(length(mod_ids) > 0){
    all_confident_mods[file_id] <- paste(mod_ids, collapse=", ")
  }
  
  consensus_pred <- length(mod_ids) >= 3 && (
    length(unique(mod_ids)) == 1)
  cat(consensus_pred, "\n")
  if(consensus_pred){
    all_consensus_files[mod_ids[1]] <- paste(all_consensus_files[mod_ids[1]],
                                             file_id, collapse = ", ")
    consensus_files_count[mod_ids[1]] <- consensus_files_count[mod_ids[1]] + 1
  }
}

View(all_confident_mods)