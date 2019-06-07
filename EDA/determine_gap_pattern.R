# Main Q: Is the gap pattern unique for all test chunks?
rm(list=ls())
library(data.table)

chunk_size <- 150000
num_test_files <- 2624
data_folder <- "/media/tom/cbd_drive/Kaggle/LANL/Data/"

# Train size: 120*(4095 + 4096*1279) - 120 cycles of length 4095 + 4096*1279 (!)
gcd <- function(a,b) ifelse (b==0, a, gcd(b, a %% b))
gcd(4095 + 4096*1279, chunk_size) # 1 - Hurray!

# Add all gap patterns to a string vector and verify the patterns are unique
num_train_chunks <- 4194
patterns <- rep(NA, num_train_chunks)
step <- 1
cycle_id <- 0
chunk_id <- 1
end_chunk_offset <- 0
gap_ids <- c()

while(chunk_id <= num_train_chunks){
  prev_step <- step
  gap_increment <- 4096 - (cycle_id==0)
  step <- ((step + gap_increment - 1) %% chunk_size) + 1
  if(step < prev_step){
    cat(chunk_id, "\n")
    patterns[chunk_id] <- paste(gap_ids, collapse="-")
    chunk_id <- chunk_id + 1
    gap_ids <- c()
    step <- step + end_chunk_offset
  }
  
  gap_ids <- c(gap_ids, step) 
  cycle_id <- (cycle_id + 1) %% 1280
}

unique_patterns <- unique(patterns)
table(as.integer(table(patterns[1:num_test_files])))
pattern_ids <- match(patterns, unique_patterns)
View(pattern_ids)
