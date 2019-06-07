shinyServer(function(input, output, session) {
  # Decrement the id of the first analysis variable
  observeEvent(input$prev_x1, {
    select_id <- max(c(1, which(input$x1 == analysis_cols_first)-1))
    updateSelectInput(session, "x1", selected=analysis_cols_first[select_id])
  })
  
  # Increment the id of the first analysis variable
  observeEvent(input$next_x1, {
    select_id <- min(c(length(analysis_cols_first),
                       which(input$x1 == analysis_cols_first)+1))
    updateSelectInput(session, "x1", selected=analysis_cols_first[select_id])
  })
  
  # Decrement the id of the second analysis variable
  observeEvent(input$prev_x2, {
    select_id <- max(c(1, which(input$x2 == analysis_cols_second)-1))
    updateSelectInput(session, "x2", selected=analysis_cols_second[select_id])
  })
  
  # Increment the id of the second analysis variable
  observeEvent(input$next_x2, {
    select_id <- min(c(length(analysis_cols_second),
                       which(input$x2 == analysis_cols_second)+1))
    updateSelectInput(session, "x2", selected=analysis_cols_second[select_id])
  })
  
  # Subset the quake data using the selected time slider
  train_time_selection <- reactive({
    start_id <- 1 + floor(input$data_start*num_data_rows)
    end_id <- floor((input$data_start+increment_period)*num_data_rows)
    out <- quake_data[start_id:end_id]
    out
  })
  
  # Plot the raw quake test time series data
  output$boxplot_mae_plotly <- renderPlotly({
    if(input$x2 == "None"){
      p <- plot_ly(x=data[[input$x1]], y=data[[target]], type="box") %>% 
        layout(xaxis=list(title=input$x1), 
               yaxis=list(title='MAE'))
    } else{
      p <- plot_ly(x=data[[input$x1]], y=data[[target]], color=data[[input$x2]],
                   type="box") %>% 
        layout(boxmode="group", 
               xaxis=list(title=input$x1), 
               yaxis=list(title='MAE'))
    }
    p
  })
})