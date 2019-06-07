shinyServer(function(input, output, session) {
  # Decrement the data start period 
  observeEvent(input$prev_period, {
    updateSliderInput(session, "data_start",
                      value=max(c(0, input$data_start-increment_period)))
  })
  
  # Increment the data start period 
  observeEvent(input$next_period, {
    updateSliderInput(session, "data_start",
                      value=min(c(1-increment_period,
                                  input$data_start+increment_period)))
  })
  
  # Subset the quake data using the selected time slider
  train_time_selection <- reactive({
    start_id <- 1 + floor(input$data_start*num_data_rows)
    end_id <- floor((input$data_start+increment_period)*num_data_rows)
    out <- quake_data[start_id:end_id]
    out
  })
  
  # Plot the raw quake training time series data
  output$quake_train_plotly <- renderPlotly({
    plot_data <- train_time_selection()
    p <- plot_ly(x=(1:nrow(plot_data)), y=plot_data$acoustic_data,
                 color=plot_data$plot_color, type = 'scatter',
                 mode ="lines") %>%
      layout(xaxis = list(title="Step"),
             yaxis = list(title="Value"))
    p
  })
  
  # Decrement the test file id
  observeEvent(input$prev_test, {
    test_file_id <- match(input$test_file, test_files)
    updateSelectInput(session, "test_file",
                      selected=test_files[max(c(1, test_file_id-1))])
  })
  
  # Increment the test file id
  observeEvent(input$next_test, {
    test_file_id <- match(input$test_file, test_files)
    updateSelectInput(session, "test_file",
                      selected=test_files[
                        min(c(num_test_files, test_file_id+1))])
  })
  
  # Load the test file segment
  test_selection <- reactive({
    out <- fread(file.path(test_folder, paste0(input$test_file, ".csv")))
    out
  })
  
  most_likely_gap_pattern_test_file <- reactive({
    test_file_id <- match(input$test_file, test_files)
    out <- most_likely_gap_patterns$most_likely_pattern[test_file_id]
    out
  })
  
  # Plot the raw quake test time series data
  output$quake_test_plotly <- renderPlotly({
    plot_data <- test_selection()
    most_likely_gap_pattern_test_file <- most_likely_gap_pattern_test_file()
    p <- plot_ly(x=(1:nrow(plot_data)), y=plot_data$acoustic_data,
                 color=plot_data$plot_color, type = 'scatter',
                 mode ="lines") %>%
      layout(xaxis = list(title="Step"),
             yaxis = list(title="Value"))
    for(gap_line_id in 1:37){
      x_line <- most_likely_gap_pattern_test_file + (gap_line_id-1)*4096 + 1
      if(x_line <= 150000){
        y_line <- max(quantile(plot_data$acoustic_data, 0.95),
                      abs(plot_data$acoustic_data[x_line]))
        p <- p %>% add_segments(x=x_line, xend=x_line, y=-1*y_line, yend=y_line,
                                color="red", showlegend=FALSE)
      }
    }
    p
  })
})