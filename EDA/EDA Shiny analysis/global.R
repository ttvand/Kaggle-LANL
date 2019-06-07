# Load required libraries
rm(list=ls())
library(shiny)
library(data.table)
library(shinythemes)
library(plotly)

quake_file <- "quake_1.rds"
drop_connect_artifacts <- FALSE

data_folder <- "/media/tom/cbd_drive/Kaggle/LANL/Data/"
quake_data <- readRDS(file.path(data_folder, quake_file))
num_data_rows <- nrow(quake_data)

# Add the identifier of large gaps
quake_data$time_gap <- c(FALSE, diff(quake_data$time_to_failure) < -1e-4)
last_of_block <- c(quake_data$time_gap[1:length(quake_data$time_gap)], FALSE)
quake_data$plot_color <- ifelse(cumsum(quake_data$time_gap) %% 2 == 0,
                                "block_1", "block_2")

if(drop_connect_artifacts){
  quake_data$acoustic_data[last_of_block] <- NA
}

# Set the random initial zoom period randomly
# increment_period <- 1e-1
increment_period <- 1e-2
# increment_period <- 1e-3
# increment_period <- 1/floor(nrow(quake_data)/150000)
num_periods <- 1/increment_period
init_rand_start <- sample(0:(num_periods-1), 1)/num_periods

# Extract the test file names
test_folder <- file.path(data_folder, "/test")
test_files <- list.files(test_folder)
test_files <- gsub(".csv$", "", test_files)
num_test_files <- length(test_files)
init_test_file <- sample(test_files, 1)

# Extract the most likely gap patterns for the test files
most_likely_gap_patterns <- fread(file.path(
  getwd(), "../../Data/Obsolete/most_likely_patterns.csv"))