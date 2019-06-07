# Main Q: How to combine unbiased predictions to minimize the MAE
# Conclusion: using the mean!
num_preds <- 6
num_sims <- 1e4
sample_size <- 1e3

predictions <- matrix(nrow=num_sims, ncol=num_preds,
                      dimnames=list(paste("Sim", 1:num_sims),
                                    paste("Prediction", 1:num_preds)))

for(sim in 1:num_sims){
  for(pred in 1:num_preds){
    x <- rnorm(sample_size)
    predictions[sim, pred] <- median(x)
  }
}

median_preds <- apply(predictions, 1, median)
mean_preds <- apply(predictions, 1, mean)
median_better <- abs(median_preds) < abs(mean_preds)
table(median_better)