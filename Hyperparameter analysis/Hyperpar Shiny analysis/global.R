# Load required libraries
rm(list=ls())
library(shiny)
library(data.table)
library(shinythemes)
library(plotly)

hyperpar_file <- c(
  "hyperpar_sweep_lgb_sequential19-05-30-17-10.csv",
  "hyperpar_sweep_lgb_sequential19-05-30-18-49.csv",
  "hyperpar_sweep_lgb_sequential19-05-30-22-41.csv",
  "hyperpar_sweep_rnn_sequential19-05-31-15-52.csv",
  "hyperpar_sweep_rnn_sequential19-05-31-17-45.csv",
  "hyperpar_sweep_rnn_sequential19-05-31-22-05.csv",
  "hyperpar_sweep_rnn_sequential19-06-01-12-10.csv",
  "hyperpar_sweep_rnn_sequential19-06-01-21-48.csv",
  "hyperpar_sweep_rnn_sequential19-06-02-11-51.csv",
  "hyperpar_sweep_lgb_sequential19-06-02-23-10.csv",
  "hyperpar_sweep_lgb_sequential19-06-03-08-26.csv",
  "hyperpar_sweep_lgb_sequential19-06-03-09-39.csv",
  "hyperpar_sweep_lgb_sequential19-06-03-10-40.csv"
)[13]

data <- fread(file.path(getwd(), "..", hyperpar_file))
target <- "MAE_normalized"
analysis_cols_first <- setdiff(colnames(data), c(target, "split"))
data[,(analysis_cols_first):= lapply(.SD, as.factor),
     .SDcols = analysis_cols_first]
first_selected <- sample(analysis_cols_first, 1)
analysis_cols_second <- c("None", analysis_cols_first)
