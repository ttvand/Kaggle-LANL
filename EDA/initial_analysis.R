rm(list=ls())
library(data.table)

data <- fread(file.path(getwd(), "Data", "train.csv"))

# Q: How many earthquakes are there in the train data?
# A: 16, of which 15 entire earthquake cycles
ttf_diff <- diff(data$time_to_failure)
quake_ids <- 1 + which(ttf_diff > 0)
num_quakes <- length(quake_ids)
cat("Num quakes:", num_quakes)

# Q: How long are the periods before the quakes?
# A: between 7 and 16 seconds (units of time in the setup)
data$time_to_failure[c(1, quake_ids)]

# Q: Are the cycle times independent?
# A: No, negative correlation of about -.5
plot(data$time_to_failure[quake_ids], ylab="Cyle duration")
plot(data$time_to_failure[head(quake_ids, -1)],
     data$time_to_failure[tail(quake_ids, -1)],
     xlab="Prev cycle", ylab="Current cycle")

# Q: Are the time steps approximately evenly spaced?
# A: No, every 4096 observations there are the equivalent of 1e6 missing
# observations. Every 1280*4096 observations, the contiguous observations only
# last 4095 steps.
# table(round(log10(abs(ttf_diff))))
large_step_ids <- which(ttf_diff < -1e-4)
table(diff(large_step_ids))
table(diff(which(diff(large_step_ids) == 4095)))
