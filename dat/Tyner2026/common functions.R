# Reproducible manuscript
{
  # Available in package form for testing purposes at https://github.com/CenterForOpenScience/reproduciblemanuscript
  # Synced on May 23, 2025
  
  # Knit placeholders to/from docx files
  # This function knits together an output .docx document from a template .docx document with placeholders and analysis code
  # template_docx_file The input .docx file containing the placeholders (in this case text in curly brackets{})
  # template_drive_ID The Google Drive ID of the template document containing the placeholders (in this case text in curly brackets{}). May be either a .docx or a Google Doc.
  # knitted_docx_file The output .docx file that knits together the figures/text/stats from the code
  # knitted_drive_ID The Google Drive ID of the output knit .docx file that will be overwritten on the knit. Note that this must point to an EXISTING .docx on google drive (can be a blank file uploaded for the purpose)
  # placeholder_object_source (optional) If NA, the code will extract placeholder objects from the current global environment. Alternatively, this can be pointed to an existing .R script to run, runs the code, and will extract the placeholders from that run environment. Finally, this could be a list (as in the output of an envirnment made into a list) and given directly.
  
  knit_docx <- function(template_docx_file = NA,
                        template_drive_ID = NA,
                        knitted_docx_file = NA,
                        knitted_drive_ID = NA,
                        placeholder_object_source = NA ){
    # Libraries
    library(officer)
    library(ggplot2)
    library(pandoc)
    
    # Authorize google docs if needed
    if(!is.na(template_drive_ID) | !is.na(knitted_drive_ID)) {
      library(googledrive)
      drive_auth()
    }
    
    # Throw an error if file paths are not specified
    if (is.na(template_docx_file) & is.na(template_drive_ID)) {
      errorCondition("You must specify a path or google drive ID for the template doc.")
    }
    if (is.na(knitted_docx_file) & is.na(knitted_drive_ID)) {
      errorCondition("You must specify a path or google drive ID for the knitted doc.")
    }
    
    # Throw an error if BOTH file paths are not specified
    if (!is.na(template_docx_file) & !is.na(template_drive_ID)) {
      errorCondition("You must choose either a path or google drive ID for the template doc, not both.")
    }
    if (!is.na(knitted_docx_file) & !is.na(knitted_drive_ID)) {
      errorCondition("You must specify either a path or google drive ID for the knitted doc, not both.")
    }
    
    # Get all objects in the specified environment
    
    if (is.list(placeholder_object_source)){
      # List of objects
      generated_objects <- placeholder_object_source
    } else if (is.na(placeholder_object_source)){
      # Unlisted (use global)
      generated_objects <- rev(as.list(globalenv()))
    } else if (endsWith(placeholder_object_source,".R")){
      # Source script
      get_environment <- function(file){
        source(file,local=TRUE)
        return(rev(as.list(environment())))
      }
      generated_objects <- get_environment(placeholder_object_source)
    } else {
      # Not found
      errorCondition("Source not identified")
    }
    
    
    # Get and prep the template doc
    # Run docx-docx conversion hack to merge .docx chunks if native .docx.
    # This step is necessary because text blocks are stored in chunks, including
    # in some instances within the same word. To make things find/replaceable, the
    # full placeholder name must be all in one chunk. "Converting" the file reorganizes
    # the text chunks, allowing placeholders to be found/replaced consistently.
    docx_out <- tempfile(fileext = ".docx")
    if (!is.na(template_drive_ID)) {
      drive_file_name <- drive_get(as_id(template_drive_ID))$name
      drive_download(file=as_id(template_drive_ID), path = docx_out)
      if (substr(drive_file_name, nchar(drive_file_name)-4, nchar(drive_file_name))==".docx"){

        pandoc::pandoc_convert(file = docx_out, from="docx", to = "docx",output=docx_out)
      }
    } else {
      # Get styles before conversion (which loses them)
      doc <- read_docx(path = template_docx_file)
      styles <- styles_info(doc)
      pandoc::pandoc_convert(file = template_docx_file, from="docx", to = "docx",output=docx_out)
    }

    doc <- read_docx(path = docx_out)
    
    # Check classes of objects so they can be set in appropriate types
    object_names <- objects(generated_objects,sorted = FALSE)
    object_classes <- do.call(c,lapply(object_names,
                                       function(obj_name) { paste(class(generated_objects[[obj_name]]),collapse=",")
                                       }))
    
    text_object_class_blacklist <- c("function","gg","ggplot","ggplot,gg","gg,ggplot","bundled_ggplot")
    pontential_text_objects <- object_names[!object_classes %in% text_object_class_blacklist]
    
    pontential_figure_objects <- object_names[object_classes == "bundled_ggplot"]
    
    # Search the generated text placeholders and replace them if found in the template document
    for (obj_name in pontential_text_objects) {
      #print(obj_name)
      doc <- cursor_begin(doc)
      if (cursor_reach_test(doc, paste0("\\{",obj_name,"\\}"))){
        tryCatch(doc <- invisible(
          body_replace_all_text(
            doc,
            old_value = paste0("{",obj_name,"}"),
            new_value = as.character(generated_objects[[obj_name]]),
            #warn=FALSE,
            fixed=TRUE
          )
        ), error=function(e) {e}, warning=function(w) {w}
        )
      }
    }
    
    # Insert figures
    for (figure_name in pontential_figure_objects) {
      # Save ggplot object to a temporary file
      png_out <- tempfile(fileext = ".png")
      ggsave(filename = png_out,
             plot = generated_objects[[figure_name]]$plot,
             height = generated_objects[[figure_name]]$height,
             width = generated_objects[[figure_name]]$width,
             units = generated_objects[[figure_name]]$units,
             bg = generated_objects[[figure_name]]$bg)
      
      # Find and replace figure placeholders
      if (cursor_reach_test(doc, paste0("\\{",figure_name,"\\}"))){
        doc <- cursor_begin(doc)
        doc <- cursor_reach(doc, paste0("\\{",figure_name,"\\}"))
        if(generated_objects[[figure_name]]$scaling=="autoheight"){
          doc <- body_add_img(x = doc, src = png_out,pos="on",
                              height=6.5*generated_objects[[figure_name]]$height/generated_objects[[figure_name]]$width,
                              width=6.5,
                              unit="in")
        } else {
          doc <- body_add_img(x = doc, src = png_out,pos="on",
                              height=generated_objects[[figure_name]]$height,
                              width=generated_objects[[figure_name]]$width,
                              unit=generated_objects[[figure_name]]$units)
        }
      }
    }
    
    # Export knitted doc
    if (!is.na(knitted_docx_file)) {
      print("Saving knitted document locally")
      print(doc, target=knitted_docx_file)
      }
    if (!is.na(knitted_drive_ID)) {
      print("Uploading knitted doc to Google Drive")
      temp_for_upload <- tempfile(fileext = ".docx")
      print(doc, target=temp_for_upload)
      drive_update(media=temp_for_upload,file=as_id(knitted_drive_ID))
    }
  }
  

  # Bundles ggplot object with dimensions
  # A wrapper function for ggplot objects that specifies the dimensions they will be in. Needed because .docx and other document types require dimensions for images
  # A ggplot plot
  # Numerical width of the ggplot # needed to insert into documents
  # Numerical height of the ggplot # needed to insert into documents
  # The units the height and width are in (defaults to "in")
  bundle_ggplot <- function(plot,width=6.5,height=4,units="in",bg="white",scaling="autoheight"){
    structure(list(plot=plot,
                   height=height,
                   width=width,
                   units=units,
                   bg=bg,
                   scaling=scaling),
              class="bundled_ggplot")
  }

  # reviews bundled ggplot object as it will appear in print
  # A convenience function so that you can preview what the bundle_ggplot object will look like in the specified dimensions
  # bundled_ggplot The bundled ggplot from the bundle_ggplot object
  # Quality of life function to show ggplot objects in their assigned scaling
  # Code adapted from https://github.com/nflverse/nflplotR/blob/main/R/ggpreview.R
  
  preview_bundled_ggplot <- function(bundled_ggplot){
    file <- tempfile()
    ggplot2::ggsave(
      file,
      plot = bundled_ggplot$plot,
      device = "png",
      scale = 1,
      width = bundled_ggplot$width,
      height = bundled_ggplot$height,
      units = bundled_ggplot$unit,
      limitsize = TRUE,
      bg = NULL
    )
    rstudioapi::viewer(file)
  }
  
  export_bundled_ggplot <- function(bundled_ggplot,file,device="png"){
    ggplot2::ggsave(
      file,
      plot = bundled_ggplot$plot,
      device = device,
      scale = 1,
      width = bundled_ggplot$width,
      height = bundled_ggplot$height,
      units = bundled_ggplot$unit,
      limitsize = TRUE,
      bg = NULL
    )
  }
}

# Aesthetic functions and presets
{
  # Charts
  {
    rounded.bars <- function(data,nesting.structure,chart.palette=NA,
                             display_axis=TRUE,axis_only=FALSE,legend=FALSE,legend.color="black",
                             flip_x = FALSE,weightvar=NA){
      # Data manipulation
      {
        data.nested <- merge(data,nesting.structure,by="cat",all.x=TRUE,all.y=FALSE)
        
        if (is.na(weightvar)){
          data.nested$weight <- 1
        } else {
          data.nested$weight<-data.nested[[weightvar]]
        }
        cats_rects <- data.nested %>%
          group_by(cat,cat2,cat3) %>%
          #summarise(count=n())%>%
          summarise(count=sum(weight))%>%
          ungroup() %>%
          arrange(match(cat, nesting.structure$cat))%>%
          mutate(proportion=count/sum(count),
                 cumproportion = cumsum(proportion))
        cats_rects$cat <- ordered(cats_rects$cat,levels=nesting.structure$cat,labels=nesting.structure$cat)
        
        if(legend==TRUE){
          cats_rects$count <- 1
          cats_rects <- cats_rects %>%
            group_by(cat2) %>%
            mutate(count2=n(),
                   proportion = 1/n()) %>%
            group_by(cat3) %>%
            mutate(count3=n()) %>%
            group_by(cat2,cat3) %>%
            mutate(count23=n())
          
          cats_rects$proportion<-cats_rects$proportion*1/length(unique(cats_rects$cat2))
          cats_rects$proportion <- ifelse(!is.na(cats_rects$cat3),
                                          cats_rects$proportion*(1/cats_rects$count3),
                                          cats_rects$proportion)
          cats_rects$proportion <- ifelse(!is.na(cats_rects$cat3),
                                          cats_rects$proportion*(1/length(unique(cats_rects$cat2))/sum(cats_rects[!is.na(cats_rects$cat3),]$proportion)),
                                          cats_rects$proportion)
          
          cats_rects$cumproportion <- cumsum(cats_rects$proportion)
        }
        
        cats_rects$xmin <- ifelse(is.na(cats_rects$cat3),
                                  cumsum(cats_rects$proportion)-cats_rects$proportion,
                                  NA)
        cats_rects$xmax <- ifelse(is.na(cats_rects$cat3),
                                  cumsum(cats_rects$proportion),
                                  NA)
        cats_rects$ymin <- ifelse(is.na(cats_rects$cat3),
                                  0,NA)
        cats_rects$ymax <- ifelse(is.na(cats_rects$cat3),
                                  1,NA)
        
        # Force last percent to be exactly equal to 1, needed due to compounding rounding error from adding floats
        cats_rects$cumproportion[nrow(cats_rects)] <- 1
        cats_rects$xmax[nrow(cats_rects)] <- 1
        
        cats_rects$yproportion <- NA
        cats_rects$ycumproportion <- NA
        cats_rects$xinternalproportion <- NA
        cats_rects$xinternalcumproportion <- NA
        for(cat2 in unique(cats_rects[!is.na(cats_rects$cat3),]$cat2)) {
          cats_rects$selected2 <- !is.na(cats_rects$cat2) & cats_rects$cat2==cat2
          xmin_total <- 
            min(cats_rects[cats_rects$selected2,]$cumproportion-cats_rects[cats_rects$selected2,]$proportion)
          
          xmax_total <- max(cats_rects[cats_rects$selected2,]$cumproportion)
          
          for(cat3 in unique(cats_rects[cats_rects$cat2==cat2,]$cat3)){
            cats_rects$selected3 <- !is.na(cats_rects$cat3) & 
              cats_rects$cat2==cat2 & 
              cats_rects$cat3==cat3
            cats_rects[cats_rects$selected3,]$yproportion <- 
              sum(cats_rects[cats_rects$selected3,]$proportion) / 
              sum(cats_rects[cats_rects$selected2,]$proportion)
            
            cats_rects[cats_rects$selected3,]$xinternalproportion <- 
              cats_rects[cats_rects$selected3,]$proportion / 
              sum(cats_rects[cats_rects$selected3,]$proportion)
            
            cats_rects[cats_rects$selected3,]$xinternalcumproportion <- 
              cumsum(cats_rects[cats_rects$selected3,]$xinternalproportion)
            
            cats_rects[cats_rects$selected3,]$xmin <-
              (cats_rects[cats_rects$selected3,]$xinternalcumproportion-cats_rects[cats_rects$selected3,]$xinternalproportion)*(xmax_total-xmin_total) +
              xmin_total
            
            cats_rects[cats_rects$selected3,]$xmax <-
              cats_rects[cats_rects$selected3,]$xinternalcumproportion*(xmax_total-cats_rects[cats_rects$selected3,]$xmin) +
              cats_rects[cats_rects$selected3,]$xmin
          }
          if(all(as.character(cats_rects$cat)==as.character(cats_rects$cat2))){
            cats_rects <- cats_rects %>%
              group_by(cat3) %>%
              mutate(ycumproportion = (row_number()==1)*yproportion) %>%
              mutate(ycumproportion = cumsum(ifelse(is.na(ycumproportion), 0, ycumproportion)) + ycumproportion*0)
          } else{
            cats_rects <- cats_rects %>%
              group_by(cat3) %>%
              mutate(ycumproportion = (row_number()==1)*yproportion) %>%
              ungroup() %>%
              mutate(ycumproportion = cumsum(ifelse(is.na(ycumproportion), 0, ycumproportion)) + ycumproportion*0)
          }
          
          cats_rects[cats_rects$selected2,]$ymin <- cats_rects[cats_rects$selected2,]$ycumproportion-cats_rects[cats_rects$selected2,]$yproportion
          cats_rects[cats_rects$selected2,]$ymax <- cats_rects[cats_rects$selected2,]$ycumproportion
        }
        
        cats_rects$xcenter <- (cats_rects$xmin+cats_rects$xmax)/2
        cats_rects$ycenter <- (cats_rects$ymin+cats_rects$ymax)/2
        
        if(flip_x == TRUE){
          cats_rects$xmax.temp <- cats_rects$xmin*-1
          cats_rects$xmin <- cats_rects$xmax*-1
          cats_rects$xmax <- cats_rects$xmax.temp
          cats_rects$xmax.temp <- NULL
          cats_rects$ymax.temp <- cats_rects$ymin*-1
          cats_rects$ymin <- cats_rects$ymax*-1
          cats_rects$ymax <- cats_rects$ymax.temp
          cats_rects$ymax.temp <- NULL
          
          cats_rects$xcenter <- cats_rects$xcenter*-1
          cats_rects$ycenter <- cats_rects$ycenter*-1
        }
        
        
      }
      
      # Chart generation
      {
        # Base chart
        {
          if (!axis_only){
            plot <- ggplot(data=cats_rects,aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,fill=cat)) + 
              funkyheatmap::geom_rounded_rect(radius=unit(3, "points"),show.legend=FALSE,
                                              color="black",
                                              size=0)+
              scale_x_continuous(expand=expansion(add = c(0, .05)),
                                 labels = scales::percent_format(),
                                 breaks=c(0,.25,.5,.75,1))
          } else {
            plot <- ggplot(data=cats_rects,aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,fill=cat)) + 
              funkyheatmap::geom_rounded_rect(radius=unit(3, "points"),show.legend=FALSE,
                                              color="black",
                                              size=0)
          }
          
          labels <- c("0%","25%","50%","75%","100%")
          breaks <- c(0,.25,.5,.75,1)
          limits <- c(0,1)
          
          if(flip_x == TRUE){
            labels <- rev(labels)
            breaks <- -1*rev(breaks)
            limits <- -1*rev(limits)
          }
          
          plot <- plot +
            scale_x_continuous(#expand=expansion(add = c(0, .05)),
              labels = labels,
              breaks=breaks,
              limits=limits)+
            theme_light() +
            theme(legend.position = "none",
                  legend.title=element_blank(),
                  panel.border = element_blank(),
                  panel.grid = element_blank(),
                  axis.title.x=element_blank(),
                  axis.title.y=element_blank(),
                  axis.text.y = element_blank(),
                  axis.line = element_blank(),
                  axis.ticks.y = element_blank(),
                  plot.margin = margin(t = 0, r = 0, b = 0, l = 0),
            )
        }
        # Options
        {
          if(display_axis==FALSE){
            plot <- plot + theme(axis.text.x=element_blank(),axis.title.x=element_blank(),axis.ticks.x = element_blank())
          }
          if (!anyNA(chart.palette)){
            plot <- plot + scale_fill_manual(values=chart.palette[levels(cats_rects$cat) %in% cats_rects$cat])
          }
          if(legend==TRUE){
            # plot <- plot+
            #   geom_text(data=cats_rects,aes(x=xcenter,y=ycenter,label=cat),
            #             color=legend.color)
          }
        }
        # Output
        {
          plot
          return(list("cats_rects"=cats_rects,"plot" = plot))
        }
      }
    }
    
    snakebins <- function(data,nesting.structure,chart.palette=NA,
                          display_axis=TRUE,axis_only=FALSE,legend.color="black",
                          n_bins=6,n_bins_max=NA,bin_width_x=1,
                          flip_x = FALSE,collapsevar=NA){
      # Data manipulation
      snake_bins <- function(n,n_bins){
        do.call(rbind,lapply(1:ceiling(n/n_bins), function (i){
          if((i %% 2) == 1){ymax <- 1:n_bins
          } else {ymax <- n_bins:1}
          xmax <- rep(i,n_bins)
          ymin <- ymax-1
          xmin <- xmax-1
          
          xmin <- xmin*bin_width_x
          xmax <- xmax*bin_width_x
          ymin <- ymin*1/n_bins
          ymax <- ymax*1/n_bins
          xcenter <- (xmin+xmax)/2
          ycenter <- (ymin+ymax)/2
          data.frame(xmin,xmax,ymin,ymax,xcenter,ycenter)
        }))[1:n,]
      }
      
      # If data are to be collapsed, collapse into upper category
      if (!is.na(collapsevar)){
        # Generate initial weights
        data$collapsevar <- data[[collapsevar]]
        data <- data %>%
          group_by(collapsevar) %>%
          mutate(weight=1/n())
        
        # Get reweighted 
        proportion <- data %>%
          group_by(cat) %>%
          #summarise(count=n())%>%
          summarise(weightcountrounded=round(sum(weight),0),
                    weightcount=sum(weight)) 
        
        # Create expanded dataset
        cats.expanded <- do.call(c,lapply(1:nrow(proportion),
                                          function(x) {
                                            rep(proportion$cat[x],proportion$weightcountrounded[x])
                                          }))
        data <- data.frame(cats.expanded)
        colnames(data) <- c("cat")
      }
      
      data.bins <- cbind(arrange(data,data$cat),snake_bins(n=nrow(data),n_bins))
      
      # Chart generation
      {
        if(!axis_only){
          snakebin_plot <- ggplot(data=data.bins,aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,fill=cat)) + 
            funkyheatmap::geom_rounded_rect(
              radius=unit(3, "points"),show.legend=FALSE,
              color="black",size=0)
          #)
        } else {
          snakebin_plot <- ggplot(data=data.bins,aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,fill=cat))+
            funkyheatmap::geom_rounded_rect(
              radius=unit(3, "points"),show.legend=FALSE,
              color="black",fill="white",size=0)
        }
        
        labels <- c("0",n_bins_max)
        breaks <- c(0,bin_width_x*n_bins_max/n_bins)
        limits <- c(0,bin_width_x*n_bins_max/n_bins)
        
        if(flip_x == TRUE){
          labels <- rev(labels)
          breaks <- -1*rev(breaks)
          limits <- -1*rev(limits)
        }
        
        snakebin_plot <- snakebin_plot + 
          scale_x_continuous(#expand=expansion(add = c(0, .05)),
            expand=c(0,2),
            #expand=expansion(add = c(10, 10)),
            labels = labels,
            breaks= breaks,
            limits = limits)+
          theme_light() +
          theme(legend.position = "none",
                legend.title=element_blank(),
                panel.border = element_blank(),
                panel.grid = element_blank(),
                axis.title.x=element_blank(),
                axis.title.y=element_blank(),
                axis.text.y = element_blank(),
                axis.line = element_blank(),
                axis.ticks.y = element_blank(),
                plot.margin = margin(t = 0, r = 0, b = 0, l = 0)
          )
        # Options
        {
          if(display_axis==FALSE){
            snakebin_plot <- snakebin_plot + theme(axis.text.x=element_blank(),axis.title.x=element_blank(),axis.ticks.x = element_blank())
          }
          if (!anyNA(chart.palette)){
            snakebin_plot <- snakebin_plot + scale_fill_manual(values=chart.palette[levels(data$cat) %in% data$cat])
          }
          }
        # Output
        {
          snakebin_plot
          return(list("plot" = snakebin_plot))
        }
      }
    }
    
    # Color palette
    {
      palette_score_charts <- c("#00a2e7",
                                "#ED1B34",
                                "#bece30",
                                "#1DBBBE",
                                "#770265",
                                "#304251",
                                "#083259",
                                "#B2023E"
      )
    }
  }
  
  # Text formatting
  {
    format.round <- function(x,digits,leading.zero=TRUE){
      out <- format(round(x,digits),nsmall=digits,trim=TRUE)
      if(leading.zero==FALSE){
        out <- ifelse(substr(out, start = 1, stop = 2)=="0.",
                      substr(out, start = 2, stop = nchar(out)),
                      out)
        out <- ifelse(substr(out, start = 1, stop = 3)=="-0.",
                      paste0("-",substr(out, start = 3, stop = nchar(out))),
                      out)
      }
      return(out)
    }
    
    format.text.CI <- function(point.estimate,CI.lb,CI.ub,alpha=.05,digits=1,
                               format.percent=FALSE,leading.zero=TRUE,
                               CI.prefix = TRUE,CI.sep=" - ",CI.bracket=c("[","]")){
      if (format.percent==TRUE){
        point.estimate <- 100*point.estimate
        CI.lb <- 100*CI.lb
        CI.ub <- 100*CI.ub
        end.notation <- "%"
      } else {
        end.notation <- ""
      }
      
      paste0(format.round(point.estimate,digits,leading.zero=leading.zero),
             end.notation," ",CI.bracket[1],
             {if(CI.prefix){ paste0(100*(1-alpha),"% CI ")} else {""}},
             format.round(CI.lb,digits,leading.zero=leading.zero),CI.sep,format.round(CI.ub,digits,leading.zero=leading.zero),
             end.notation,CI.bracket[2])
    }
    
    format.text.percent <- function(x,n,alpha=.05,digits=1,confint=TRUE,leading.zero=TRUE,
                                    CI.prefix=TRUE,CI.sep=" - ",CI.bracket=c("[","]")){
      p <- binconf(x,n)
      text <- paste0(format.round(100*p[1],digits),"%")
      if(confint){
        text <- format.text.CI(p[1],p[2],p[3],alpha,digits,format.percent = TRUE,
                               leading.zero=leading.zero,CI.prefix=CI.prefix,CI.sep=CI.sep,CI.bracket=CI.bracket)
      } else
        text <- paste0(format.round(100*p[1],digits=digits,leading.zero=leading.zero),"%")
      text
    }
  }
}

# Statistical functions
{
  bootstrap.clust <- function(data=NA,FUN=NA,keepvars=NA,clustervar=NA,
                              alpha=.05,tails="two-tailed",iters=200,
                              parallel=FALSE,progressbar=FALSE,
                              format.percent=FALSE,digits=1,leading.zero=TRUE,na.rm=FALSE,
                              CI.prefix=TRUE,CI.sep=" - ",CI.bracket=c("[","]")){
    # Drop any variables from the dataframe that are not required for speed (optional)
    # and/or with missing values
    data.internal <- data
    if (!anyNA(keepvars)){
      keepvars <- na.omit(unique(c(keepvars,clustervar)))
      data.internal <- data.internal[c(keepvars)]
      # Drop any rows with missing data
      if (na.rm){ data.internal <- na.omit(data.internal)}
    }
    # Warn that na.rm not enabled if keepvars isn't specified
    if (anyNA(keepvars) & na.rm==TRUE){
      warning("na.rm not enabled for boostrap.clust if keepvars isn't specified")
    }
    
    # Set up cluster designations
      if(anyNA(clustervar)){ data.internal$cluster.id <- 1:nrow(data.internal) 
      } else { data.internal$cluster.id <- data.internal[[clustervar]] }
      cluster.set <- unique(data.internal$cluster.id)
    # Generate original target variable
      point.estimate <- FUN(data.internal)
    # Create distribution of bootstrapped samples
      if (parallel==FALSE & progressbar==FALSE) {
          estimates.bootstrapped <- replicate(iters,{
          # Generate sample of clusters to include
            clust.list <- sample(cluster.set,length(cluster.set),replace = TRUE)
          # Build dataset from cluster list
            data.clust <- sapply(clust.list, function(x) which(data.internal[,"cluster.id"]==x))
            data.clust <- data.internal[unlist(data.clust),]
          # Run function on new data
            tryCatch(FUN(data.clust),finally=NA)
        },simplify=TRUE)
      } else if (parallel==TRUE & progressbar==FALSE) {
        #library(parallel)
        
        estimates.bootstrapped <- do.call(c,parallel::mclapply(1:iters,FUN=function(i){
          # Generate sample of clusters to include
          clust.list <- sample(cluster.set,length(cluster.set),replace = TRUE)
          # Build dataset from cluster list
          data.clust <- sapply(clust.list, function(x) which(data.internal[,"cluster.id"]==x))
          data.clust <- data.internal[unlist(data.clust),]
          # Run function on new data
          tryCatch(FUN(data.clust),finally=NA)
        }))
      } else if (parallel==FALSE & progressbar==TRUE) {
        library(pbapply)
        
        estimates.bootstrapped <- pbreplicate(iters,{
          # Generate sample of clusters to include
          clust.list <- sample(cluster.set,length(cluster.set),replace = TRUE)
          # Build dataset from cluster list
          data.clust <- sapply(clust.list, function(x) which(data.internal[,"cluster.id"]==x))
          data.clust <- data.internal[unlist(data.clust),]
          # Run function on new data
          tryCatch(FUN(data.clust),finally=NA)
        },simplify=TRUE,cl=detectCores()-1)
      } else if (parallel==TRUE & progressbar==TRUE) {
        #library(parallel)
        library(pbapply)
        
        estimates.bootstrapped <- do.call(c,pblapply(1:iters,FUN=function(i){
          # Generate sample of clusters to include
          clust.list <- sample(cluster.set,length(cluster.set),replace = TRUE)
          # Build dataset from cluster list
          data.clust <- sapply(clust.list, function(x) which(data.internal[,"cluster.id"]==x))
          data.clust <- data.internal[unlist(data.clust),]
          # Run function on new data
          tryCatch(FUN(data.clust),finally=NA)
        },cl=detectCores()-1))
      }
    # Generate outcomes measures
      if(is.matrix(estimates.bootstrapped)){
        n_estimates <- nrow(estimates.bootstrapped)
        SE <- unlist(lapply(1:n_estimates,function(i) {sd(estimates.bootstrapped[i,],na.rm = TRUE)}))
        if (tails == "two-tailed"){
          CI.lb <- unlist(lapply(1:n_estimates,function(i) {quantile(estimates.bootstrapped[i,], alpha/2,na.rm = TRUE)}))
          CI.ub <- unlist(lapply(1:n_estimates,function(i) {quantile(estimates.bootstrapped[i,], 1-alpha/2,na.rm = TRUE)}))
        } else if (tails == "one-tailed, upper"){
          CI.lb <- unlist(lapply(1:n_estimates,function(i) {NA}))
          CI.ub <- unlist(lapply(1:n_estimates,function(i) {quantile(estimates.bootstrapped[i,], 1-alpha,na.rm = TRUE)}))
        } else if (tails == "one-tailed, lower"){
          CI.lb <- unlist(lapply(1:n_estimates,function(i) {quantile(estimates.bootstrapped[i,], alpha,na.rm = TRUE)}))
          CI.ub <- unlist(lapply(1:n_estimates,function(i) {NA}))
        }
      } else {
        SE <- sd(estimates.bootstrapped,na.rm = TRUE)
        if (tails == "two-tailed"){
          CI.lb <- quantile(estimates.bootstrapped, alpha/2,na.rm = TRUE)
          CI.ub <- quantile(estimates.bootstrapped, 1-alpha/2,na.rm = TRUE)
        } else if (tails == "one-tailed, upper"){
          CI.lb <- NA
          CI.ub <- quantile(estimates.bootstrapped, 1-alpha,na.rm = TRUE)
        } else if (tails == "one-tailed, lower"){
          CI.lb <- quantile(estimates.bootstrapped, alpha,na.rm = TRUE)
          CI.ub <- NA
        }
      }
    
    # Outputs
      output.list <- list("point.estimate"=point.estimate,"SE"=SE,
                          "CI.lb"=CI.lb,"CI.ub"=CI.ub,
                          "estimates.bootstrapped"=estimates.bootstrapped,
                          "alpha"=alpha, "tails"=tails
                          )
      output.list[["formatted.text"]] <- 
        format.text.CI(point.estimate=point.estimate,
                       "CI.lb"=CI.lb,"CI.ub"=CI.ub,
                       alpha=alpha,digits=digits,leading.zero=leading.zero,
                       format.percent=format.percent,
                       CI.prefix=CI.prefix,CI.sep=CI.sep,CI.bracket=CI.bracket)
    return(output.list)
  }
  
  # Probability/Risk ratio
  probability.ratio <- function(data=NA,exposure,outcome,weight=NA){
    # Generate dataset
      if (is.na(data)){
        exposure <- as.numeric(exposure)
        outcome <- as.numeric(outcome)
        data <- data.frame(exposure,outcome)
        if(anyNA(weight)){
         data$weight <- 1
        } else {
          data$weight <- weight
        }
      } else {
        data$expsoure <- as.numeric(data[[exposure]])
        data$outcome <- as.numeric(data[[exposure]])
        data$weight <- as.numeric(data[[weight]])
      }
    
      prob.exposed <- weighted.mean(data[data$exposure==1,]$outcome,data[data$exposure==1,]$weight,na.rm = TRUE)
      prob.unexposed <- weighted.mean(data[data$exposure==0,]$outcome,data[data$exposure==0,]$weight,na.rm = TRUE)
      probability.ratio <- prob.exposed/prob.unexposed

    return(probability.ratio)
  }
  
  weighted.quantile <- function(x,weights,quantile=.5,na.rm=FALSE){
    # Modified from R package Limma, Gordon Smyth<smyth@wehi.edu.au>, http://bioinf.wehi.edu.au/limma
    # Check / modify data
      if (missing(weights)) 
        weights <- rep.int(1, length(x))
      else {
        if(length(weights) != length(x)) stop("'x' and 'weights' must have the same length")
        if(any(is.na(weights))) stop("NA weights not allowed")
        if(any(weights<0)) stop("Negative weights not allowed")
      }
      if(na.rm==TRUE){
        weights <- weights[i <- !is.na(x)]
        x <- x[i]
      }
      if(all(weights==0)) {
        warning("All weights are zero")
        return(NA)
      }
    # Find quantile
      order.x <- order(x)
      x <- x[order.x]
      weights <- weights[order.x]
      p <- cumsum(weights)/sum(weights)
      n <- sum(p<quantile)
      if(p[n+1] > quantile)
        x[n+1]
      else
        (x[n+1]+x[n+2])/2
  }
  
  # Weighted median, wrapper function
  weighted.median <- function (x, weight, na.rm = FALSE){
    weighted.quantile(x,weight,quantile=.5,na.rm)
  }
  
  # Clustered/weighted proportion (convenience function)
  cw.proportion <- function (data, weights, clusters,iters){

    df <- data.frame(data,weights,clusters)
    
    bootstrap.clust(data=df,
                    FUN=function(x) {
                      sum(as.numeric(x$data)*x$weights)/sum(x$weights)
                    },
                    clustervar = "clusters",
                    keepvars=c("clusters","data","weights"),
                    alpha=.05,tails="two-tailed",iters=iters,
                    format.percent=TRUE,digits=1
    )
  }
}
