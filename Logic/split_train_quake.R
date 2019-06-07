library(data.table)

data_folder <- "/media/tom/cbd_drive/Kaggle/LANL/Data/"
data <- fread(file.path(data_folder, "train.csv"))

# Get the ids of new quake cycles 
ttf_diff <- diff(data$time_to_failure)
period_starts <- c(1, 1 + which(ttf_diff > 0))
period_ends <- c(tail(period_starts, -1) - 1, nrow(data))

# Create an rds file for each quake period
num_periods <- length(period_ends)
for(i in 1:num_periods){
  quake_period <- data[period_starts[i]:period_ends[i]]
  saveRDS(quake_period, file.path(getwd(), "Data", paste0("quake_", i, ".rds")))
}
