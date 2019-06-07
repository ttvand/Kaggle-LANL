# Main Q: Is the claim that the gap of missing data is 12 micro seconds
# verifiable? Compare distributions to assess the claim.

# Current working hypothesis: the first observation of a period is not
# reliable as the value is typically lower than other values in a period
# Approach: compare gap changes versus  
rm(list=ls())
library(data.table)

quake_file <- "quake_7.rds"
data <- readRDS(file.path(getwd(), "Data", quake_file))

# Obtain the ids of gaps in the data
ttf_diff <- diff(data$time_to_failure)
large_step_ids <- which(ttf_diff < -1e-4)

# Function to obtain all diffs of uninterrupted data, skipping time gaps
get_diffs_uninterrupted <- function(signal, gap_ids, step_size){
  n <- length(signal)
  start_ids <- 1:(n-step_size)
  drop_ids <- 1 + rep(gap_ids, each=step_size) -
    rep(1:step_size, length(gap_ids)) 
  valid_drops <- drop_ids[drop_ids > 0 & drop_ids <= (n-step_size)]
  valid_start_ids <- start_ids[-valid_drops]
  return (signal[valid_start_ids+step_size] - signal[valid_start_ids])
}

# At 4M Hz, 12 micro seconds are 48 observations
observation_gap <- 48
uninterrupted_diffs <- get_diffs_uninterrupted(
  data$acoustic_data, large_step_ids, observation_gap)
gap_diffs <- data$acoustic_data[large_step_ids+1] - 
  data$acoustic_data[large_step_ids]
gap_diffs_shifted <- data$acoustic_data[large_step_ids+1+observation_gap] - 
  data$acoustic_data[large_step_ids+1]
# gap_diffs_shifted <- data$acoustic_data[large_step_ids] - 
#   data$acoustic_data[large_step_ids-observation_gap]
gap_diffs_shifted <- gap_diffs_shifted[!is.na(gap_diffs_shifted)]

plot_xlim <- 20
uninterrupted_plot_diffs <- uninterrupted_diffs[
  abs(uninterrupted_diffs)<plot_xlim]
par(mfrow=c(2, 1))
hist(uninterrupted_plot_diffs, 40, col="grey", main=paste(
  "Uninterrupted fraction diff less than 10:",
  round(mean(abs(uninterrupted_diffs) < 10), 3)))
gap_plot_diffs <- gap_diffs[abs(gap_diffs)<plot_xlim]
hist(gap_plot_diffs, 40, col="grey", main=paste(
  "Gap fraction diff less than 10:",
  round(mean(abs(gap_diffs) < 10), 3)))
par(mfrow=c(1, 1))
gap_plot_diffs_shifted <- gap_diffs_shifted[abs(gap_diffs_shifted)<plot_xlim]
hist(gap_plot_diffs_shifted, 40, col="grey", main=paste(
  "Gap fraction diff less than 10:",
  round(mean(abs(gap_diffs_shifted) < 10), 3)))

# Plot the uninterrupted fraction diff less than a limit as a function of
# the observation gap
observation_gaps <- 1:80
num_observation_gaps <- length(observation_gaps)
limit_change_threshold <- 10
fraction_below_changes <- rep(NA, num_observation_gaps)
fraction_increased <- rep(NA, num_observation_gaps)
for (gap_id in 1:num_observation_gaps){
  cat(gap_id)
  gap <- observation_gaps[gap_id]
  diffs <- get_diffs_uninterrupted(data$acoustic_data, large_step_ids, gap)
  fraction_below_changes[gap_id] <- mean(abs(diffs) <= limit_change_threshold)
  fraction_increased[gap_id] <- mean(diffs[diffs != 0] > 0)
}
gap_fraction <- mean(abs(gap_diffs) <= limit_change_threshold)
plot(fraction_below_changes)
abline(h=gap_fraction, col="red")
abline(v=observation_gap, col="green")

# Plot the values before and after the gap
plot(data$acoustic_data[large_step_ids], data$acoustic_data[large_step_ids+1],
     xlim=c(-20, 20), ylim=c(-20, 20))
abline(0, 1, col="blue")
mean(data$acoustic_data[large_step_ids+1]-data$acoustic_data[large_step_ids])
median(data$acoustic_data[large_step_ids+1]-data$acoustic_data[large_step_ids])

# Plot the average change between the last observation before the gaps and
# observations after the gap
observation_gaps <- 1:4095
num_observation_gaps <- length(observation_gaps)
limit_change_threshold <- 10
fraction_below_changes <- rep(NA, num_observation_gaps)
for (gap_id in 1:num_observation_gaps){
  gap <- observation_gaps[gap_id]
  diffs <- data$acoustic_data[large_step_ids+gap] - 
    data$acoustic_data[large_step_ids]
  fraction_below_changes[gap_id] <- mean(abs(diffs) <= limit_change_threshold)
  fraction_increased[gap_id] <- mean(diffs[diffs != 0] > 0)
}
gap_fraction <- mean(abs(gap_diffs) <= limit_change_threshold)
plot(fraction_below_changes)
abline(h=gap_fraction, col="red")

# # Verify the mean absolute positive shift as a function of the period
# # Conclusion: the data typically shifts up at the beginning of the next period
# gap_increments <- c(1:100, 4095-observation_gap - rev(1:100))
# num_gap_increments <- length(gap_increments)
# fraction_increased <- rep(NA, num_gap_increments)
# for (gap_id in 1:num_gap_increments){
#   gap_increment <- gap_increments[gap_id]
#   diffs <- data$acoustic_data[large_step_ids+gap_increment+observation_gap] - 
#     data$acoustic_data[large_step_ids+gap_increment]
#   diffs <- diffs[!is.na(diffs)]
#   fraction_increased[gap_id] <- mean(diffs[diffs!=0] > 0)
# }
# plot(fraction_increased)
# 
# # Inspect the fraction of value increments versus the start of the period
# # About 70 % goes up compared to the first measurement!
# period_increments <- 1:100
# num_period_increments <- length(period_increments)
# fraction_increased <- rep(NA, num_period_increments)
# for (incr_id in 1:num_period_increments){
#   increment <- period_increments[incr_id]
#   diffs <- data$acoustic_data[large_step_ids+increment+1] - 
#     data$acoustic_data[large_step_ids+1]
#   diffs <- diffs[!is.na(diffs)]
#   fraction_increased[incr_id] <- mean(diffs[diffs!=0] > 0)
# }
# plot(fraction_increased)