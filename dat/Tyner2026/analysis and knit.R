# Run knit_manuscript() to
# 1) Generate all placeholder stats
# 2) Generate all placeholder figures
# 3) Read the template MS Word doc
# 4) Replace all placeholder stats and figures and knit them together into a
# knitted Microsoft Word document

# The resulting objects (results_tagged_stats and results_figures) contain
# listed outputs for all analysis tags contained within the publication,
# accessed with $ (e.g. results_figures$figure_1)

# Note: packages are current CRAN versions as of Oct 27 (see readme for package
# version information)

knit_manuscript <- function(template_docx_file="template manuscript.docx",
                            template_drive_ID=NA,
                            knitted_docx_file = "knitted manuscript.docx",
                            knitted_drive_ID=NA)
{
  source("common functions.R")
  set.seed(1565079)
  # Place all numeric and character tagged placeholders into the current function environment
  print("Generating placeholder text stats")
  results_placeholder_stats <<- placeholder_stats(iters = 1000)
  print("Generating figures")
  results_figures <<- figures(iters = 1000)
  
  results_all <- append(results_placeholder_stats, results_figures)
  
  print("Knitting manuscript from template")
  knit_docx(template_docx_file=template_docx_file,
            template_drive_ID=template_drive_ID,
            knitted_docx_file = knitted_docx_file,
            knitted_drive_ID=knitted_drive_ID,
            placeholder_object_source = results_all)
}

# Create an object that contains the tagged stats
placeholder_stats <- function(iters = 100,generate_binary_outcomes_data=FALSE){

  # Load libraries
  {
    library(dplyr)
    library(ggplot2)
    library(ggExtra)
    library(DT)
    library(tidyr)
    library(stringr)
    library(Hmisc)
    library(scales)
    library(wCorr)
    library(corrplot)
    library(cowplot)
    library(ggridges) #note: using the github version, as the CRAN hasn't been pushed to get the weights functionality
    library(ggside)
    library(weights)
    library(glue)
    
  }
  
  # Load data and common functions
  {
    load("analyst data.RData")
    source("common functions.R")
  }
    
  # Data preparation
  {
    # Set defaults for convenience
    {
      if (!exists("iters")){ iters <- 100}
      if (!exists("generate_binary_outcomes_data")){ generate_binary_outcomes_data <- FALSE}
    }
    
    # Generate binary data. Note: generate_binary_outcomes_data flag by default is set 
    # to FALSE, which pulls the binary data precalculated from the data pipeline.
    # If set to TRUE, the code will run to generate the binary success measures
    # from scratch. This is time consuming, but will produce identical results)
    {
      if(generate_binary_outcomes_data==TRUE ){
        # Overwrite static version from scratch
        library(metafor)
        library(pwr)
        library(purrr)
        library(BFF)
        library(BayesFactor)
        library(effectsize)
        library(BayesRep)
        library(metaBMA)
        library(ReplicationSuccess)
        repli_binary <- calculate_repli_binary(repli_outcomes, 
                                           orig_outcomes, 
                                           paper_metadata)
      } else {
        repli_binary_orig <- repli_binary
      }
      
    }
    
    # Trim out non-version of record lines and non-COVID entries
    {
      repli_outcomes_orig <- repli_outcomes
      
      repli_outcomes <- repli_outcomes[repli_outcomes$repli_version_of_record,]
      repli_outcomes <- repli_outcomes[!repli_outcomes$is_covid,]
    }
    
    # Get and trim paper metadata information
    {
      paper_metadata_orig <- paper_metadata
      paper_metadata <- paper_metadata[c("paper_id","publication_standard","COS_pub_category","pub_year","COS_pub_expanded")]
      paper_metadata$field <- str_to_title(paper_metadata$COS_pub_category)
      paper_metadata$field <- str_replace(paper_metadata$field,"And","and")
      
      orig_outcomes <- merge(orig_outcomes,paper_metadata,by="paper_id",all.x = TRUE,all.y=FALSE)
      
      repli_outcomes_merged <- merge(repli_outcomes,orig_outcomes[,!(names(orig_outcomes) %in% c("paper_id"))],
                                     by="claim_id",all.x=TRUE,all.y=FALSE)
      orig_outcomes <- orig_outcomes %>% group_by(paper_id) %>% mutate(weight = 1/n()) %>% ungroup()
      repli_outcomes <- repli_outcomes %>% group_by(paper_id) %>% mutate(weight = 1/n()) %>% ungroup()
      repli_outcomes_merged <- repli_outcomes_merged %>% group_by(paper_id) %>% mutate(weight = 1/n()) %>% ungroup()
    }
    
    # Add variables to repli_binary
    {
      repli_binary <- merge(repli_outcomes[c("paper_id","claim_id","report_id","repli_score_criteria_met")],
                            repli_binary,all.x = FALSE,all.y=TRUE,by="report_id")
    }
    
    # Generate names for binary measures
    {
      binvars.raw <- c("repli_score_criteria_met",
                       "repli_binary_analyst",
                       "repli_binary_orig_wthn",
                       "repli_binary_rep_wthn",
                       "repli_binary_wthn_pi",
                       "repli_binary_meta_success",
                       "repli_binary_sum_p",
                       "repli_binary_telescopes",
                       "repli_binary_bf_result",
                       "repli_binary_bayes_rep",
                       "repli_binary_bma_result",
                       "repli_binary_skep_p",
                       "repli_binary_correspondence")
      binvars.order <- binvars.raw
      
      binvars.full.text <- c("Statistical signifance + pattern",
                         "Subjective interpretation",
                         "Original effect in replication's CI",
                         "Replication effect in original's CI",
                         "Replication effect in prediction interval",
                         "Meta-analysis",
                         "Sum of p-values",
                         "Small telescopes",
                         "Bayes factor",
                         "Replication Bayes factor",
                         "Bayesian meta-analysis",
                         "Skeptical p-value",
                         "Correspondence test")
      binvars.group <- c("Statistic significance / p-value based measures",
                         "SCORE specific",
                         "Contained within interval measures",
                         "Contained within interval measures",
                         "Contained within interval measures",
                         "Meta analysis based measures",
                         "Statistic significance / p-value based measures",
                         "Combined effect magnitude and uncertainty measures",
                         "Combined effect magnitude and uncertainty measures",
                         "Combined effect magnitude and uncertainty measures",
                         "Meta analysis based measures",
                         "Statistic significance / p-value based measures",
                         "Combined effect magnitude and uncertainty measures")
      
      binvars.order <- c("repli_binary_analyst",
                         "repli_score_criteria_met",
                         "repli_binary_sum_p",
                         "repli_binary_skep_p",
                         "repli_binary_orig_wthn",
                         "repli_binary_rep_wthn",
                         "repli_binary_wthn_pi",
                         "repli_binary_meta_success",
                         "repli_binary_bma_result",
                         "repli_binary_telescopes",
                         "repli_binary_bf_result",
                         "repli_binary_bayes_rep",
                         "repli_binary_correspondence")
      
      df.binvars <- data.frame(binvars.raw,binvars.full.text,binvars.group,binvars.order)
      df.binvars <- do.call(rbind,lapply(binvars.order,function(x) {
        df.binvars[df.binvars$binvars.raw==x,]
      }))
    }
    
    # Generate convenience dataset for publications
    {
      publications <- repli_outcomes_merged %>%
        select(publication_standard,COS_pub_category,COS_pub_expanded,field) %>%
        distinct()
    }
  }

  # Main section stats
  {
    # Tables
    {
      # Table 1
      {
        fields.order <- sort(c("Psychology and Health","Business","Sociology and Criminology",
                          "Economics and Finance","Political Science","Education"))
        
        format.row <- function(data){
          data <- data %>% 
            select(paper_id) %>% 
            left_join(paper_metadata %>% select(paper_id, field), by = "paper_id") %>% 
            count(field)
          total <- sum(data$n)
          field <- "Total"
          n <- paste0(total," (100%)")
          data$n <- paste0(data$n," (",
                           format.round(100*data$n/sum(data$n),1),
                           "%)")
          data <- rbind(data,data.frame(field,n))
          colnames <- data$field
          data <- data.frame(t(rbind(data,data.frame(field,n))[,-1]))
          colnames(data) <- colnames
          data[c(fields.order,"Total")]
        }
        
        p.row <- function(data){
          data <- data %>% 
            select(paper_id) %>% 
            left_join(paper_metadata %>% select(paper_id, field), by = "paper_id") %>% 
            count(field)
          total <- sum(data$n)
          field <- "Total"
          n <- 1
          data$n <- data$n/sum(data$n)
          data <- rbind(data,data.frame(field,n))
          colnames <- data$field
          data <- data.frame(t(rbind(data,data.frame(field,n))[,-1]))
          colnames(data) <- colnames
          data[c(fields.order,"Total")]
        }
        
        n.row <- function(data){
          data <- data %>% 
            select(paper_id) %>% 
            left_join(paper_metadata %>% select(paper_id, field), by = "paper_id") %>% 
            count(field)
          total <- sum(data$n)
          field <- "Total"
          n <- total
          data <- rbind(data,data.frame(field,n))
          colnames <- data$field
          data <- data.frame(t(rbind(data,data.frame(field,n))[,-1]))
          colnames(data) <- colnames
          data[c(fields.order,"Total")]
        }
        
        r1.data <- paper_status_tracking %>% filter(p1_delivery | p2_delivery)
        r2.data <- paper_status_tracking %>% filter(RR | p2_delivery)
        r3.data <- paper_status_tracking %>% filter(bushel)
        r4.data <- paper_status_tracking %>% filter((p2_delivery | RR) & !bushel)
        r5.data <- rr_internal_tracking %>% 
          filter(str_detect(type, "Replication")) %>%
          select(paper_id) %>% 
          distinct() %>% 
          semi_join(paper_status_tracking %>% filter(p1_delivery | p2_delivery), by = "paper_id")
        r6.data <-  repli_outcomes_orig %>% 
          filter(repli_type != "original and secondary data") %>% 
          filter(!is_covid) %>% 
          #filter(repli_version_of_record) %>% 
          select(paper_id) %>% 
          distinct()
        # r7.data <- repli_outcomes_orig %>%
        #   filter(repli_type != "original and secondary data") %>% 
        #   filter(!is_covid) %>% 
        #   filter(repli_version_of_record) %>% 
        #   filter(is.na(manylabs_type) | manylabs_type != "aggregation") %>% 
        #   select(paper_id, claim_id, rr_id) %>% 
        #   distinct() 
        
        r7.data <- repli_outcomes_orig %>%
          filter(repli_type != "original and secondary data") %>%
          semi_join(paper_status_tracking %>% filter(p1_delivery|p2_delivery), by = "paper_id") %>%
          filter(is.na(manylabs_type) | manylabs_type != "aggregation") %>%
          select(paper_id, claim_id, rr_id) %>%
          distinct()
        r8.data <-  repli_outcomes_orig %>%
          filter(repli_type != "original and secondary data") %>% 
          filter(!is_covid) %>% 
          filter(repli_version_of_record) %>% 
          select(paper_id, claim_id) %>% 
          distinct()
        
        for (i in 1:8){
          assign(paste0("r",i),format.row(get(paste0("r",i,".data"))))
          assign(paste0("r",i,".n"),n.row(get(paste0("r",i,".data"))))
          assign(paste0("r",i,".p"),p.row(get(paste0("r",i,".data"))))
          assign(paste0("r",i,".p.formatted"),
                 paste0(format.round(100*p.row(get(paste0("r",i,".data"))),0),"%")
                 )
        }
        
        table_1 <- rbind(r1,r2,r3,r4,r5,r6,r7,r8)
        table_1_n <- rbind(r1.n,r2.n,r3.n,r4.n,r5.n,r6.n,r7.n,r8.n)
        table_1_p <- rbind(r1.p,r2.p,r3.p,r4.p,r5.p,r6.p,r7.p,r8.p)
        table_1_p_formatted <- rbind(r1.p.formatted,r2.p.formatted,r3.p.formatted,r4.p.formatted,r5.p.formatted,r6.p.formatted,r7.p.formatted,r8.p.formatted)
        for (row in 1:nrow(table_1)){
          for (col in 1:ncol(table_1)){
            assign(paste0("table_1_",row,"_",col),
                   table_1[row,col])
            assign(paste0("table_1_p_",row,"_",col),
                   table_1_p[row,col])
            assign(paste0("table_1_p_formatted_",row,"_",col),
                   table_1_p_formatted[row,col])
          }
        }
        rm(r1,r2,r3,r4,r5,r6,r7,r8,
           r1.p,r2.p,r3.p,r4.p,r5.p,r6.p,r7.p,r8.p,
           r1.data,r2.data,r3.data,r4.data,r5.data,r6.data,r7.data,r8.data,
           r1.p.formatted,r2.p.formatted,r3.p.formatted,r4.p.formatted,r5.p.formatted,r6.p.formatted,r7.p.formatted,r8.p.formatted)
      }
      
      # Table 2
      {
        repli_effects <- repli_outcomes %>% 
          filter(!is_covid) %>% 
          filter(repli_version_of_record) %>% 
          select(report_id, claim_id, repli_sample_size_value, success = repli_score_criteria_met, repli_type,
                 repli_conv_r, repli_conv_r_lb, repli_conv_r_ub, repli_effect_size_type, repli_effect_size_value) %>%
          mutate(ser_effect = ifelse(str_detect(repli_effect_size_type, "ser_"), abs(repli_effect_size_value), NA)) %>% 
          mutate(across(contains("conv"), abs))
        
        orig_effects <- orig_outcomes %>% 
          select(claim_id, paper_id,orig_sample_size_value, orig_conv_r, orig_conv_r_lb, orig_conv_r_ub,
                 orig_effect_size_type_repli, orig_effect_size_value_repli) %>%
          semi_join(repli_effects, by = "claim_id") %>% 
          mutate(ser_effect = ifelse(str_detect(orig_effect_size_type_repli, "ser_"), abs(orig_effect_size_value_repli), NA)) %>% 
          mutate(across(contains("conv"), abs))
        
        effects_combined <- merge(repli_effects,orig_effects,by="claim_id",all.x = TRUE,all.y = FALSE)
        
        effects_combined <- effects_combined %>%
          filter(!is.na(repli_conv_r)) %>%
          filter(!is.na(orig_conv_r)) %>%
          group_by(paper_id) %>%
          mutate(weight = 1/n())
          
        
        # Number of papers and claims
        r1 <- c(length(unique(effects_combined$paper_id)),length(unique(effects_combined$paper_id)),
                length(unique(effects_combined$claim_id)),length(unique(effects_combined$claim_id)))
        
        # Number of effect sizes
        r2 <- c(sum(as.numeric(!is.na(effects_combined$orig_conv_r))*effects_combined$weight),
                sum(as.numeric(!is.na(effects_combined$repli_conv_r))*effects_combined$weight),
                sum(!is.na(effects_combined$orig_conv_r)),
                sum(!is.na(effects_combined$repli_conv_r))
        )
        
        # Median/IQR of sample sizes
        r3.median <- c(weighted.quantile(effects_combined$orig_sample_size_value,effects_combined$weight,.5),
                       weighted.quantile(effects_combined$repli_sample_size_value,effects_combined$weight,.5),
                       quantile(effects_combined$orig_sample_size_value,.5,na.rm=TRUE),
                       quantile(effects_combined$repli_sample_size_value,.5,na.rm=TRUE)
        )
        
        r3.IQR <- c(format.round(weighted.quantile(effects_combined$orig_sample_size_value,effects_combined$weight,.50) - weighted.quantile(effects_combined$orig_sample_size_value,effects_combined$weight,.25),1),
                    format.round(weighted.quantile(effects_combined$repli_sample_size_value,effects_combined$weight,.50) - weighted.quantile(effects_combined$repli_sample_size_value,effects_combined$weight,.25),1),
                    format.round(IQR(effects_combined$orig_sample_size_value,na.rm=TRUE),1),
                    format.round(IQR(as.integer(effects_combined$repli_sample_size_value),na.rm=TRUE),1)
        )

        r3 <- paste0(r3.median,"\n[",r3.IQR,"]")
        
        # Pearson's R effect size
        r4.median <- c(weighted.quantile(effects_combined$orig_conv_r,effects_combined$weight,.5),
                weighted.quantile(effects_combined$repli_conv_r,effects_combined$weight,.5),
                quantile(effects_combined$orig_conv_r,.5),
                quantile(effects_combined$repli_conv_r,.5)
        )
        
        r4.median.formatted <- format.round(r4.median,2)
        
        r4.SD <- c(sqrt(wtd.var(effects_combined$orig_conv_r,effects_combined$weight)),
                sqrt(wtd.var(effects_combined$repli_conv_r,effects_combined$weight)),
                sd(effects_combined$orig_conv_r,1),
                sd(effects_combined$repli_conv_r,1)
        )
        r4.SD.formatted <- format.round(r4.SD,2)
        
        r4 <- paste0(r4.median.formatted,"\n(",r4.SD.formatted,")")
        
        table_2 <- rbind(r1,r2,r3,r4)
        for (row in 1:nrow(table_2)){
          for (col in 1:ncol(table_2)){
            assign(paste0("table_2_",row,"_",col),
                   table_2[row,col])
            if(row==4){
              assign(paste0("table_2_median_",row,"_",col),
                     r4.median[col])
              assign(paste0("table_2_SD_",row,"_",col),
                     r4.SD[col])
              assign(paste0("table_2_median_formatted_",row,"_",col),
                     r4.median.formatted[col])
              assign(paste0("table_2_SD_formatted_",row,"_",col),
                     r4.SD.formatted[col])
            }
          }
        }
        rm(r1,r2,r3,r4)
      }
      
      # Table 3
      {
        fields.order <- c("business","economics and finance","education","political science",
                          "psychology and health","sociology and criminology")

        repli_effects <- repli_outcomes %>%
          filter(!is_covid) %>%
          filter(repli_version_of_record) %>%
          select(report_id, claim_id,repli_sample_size_value, success = repli_score_criteria_met, repli_type,
                 repli_conv_r, repli_conv_r_lb, repli_conv_r_ub, repli_effect_size_type, repli_effect_size_value) %>%
          mutate(ser_effect = ifelse(str_detect(repli_effect_size_type, "ser_"), abs(repli_effect_size_value), NA)) %>%
          mutate(across(contains("conv"), abs))

        orig_effects <- orig_outcomes %>%
          semi_join(repli_effects, by = "claim_id") %>%
          select(claim_id, paper_id,orig_sample_size_value, orig_conv_r, orig_conv_r_lb, orig_conv_r_ub,
                 orig_effect_size_type_repli, orig_effect_size_value_repli) %>%
          mutate(ser_effect = ifelse(str_detect(orig_effect_size_type_repli, "ser_"), abs(orig_effect_size_value_repli), NA)) %>%
          mutate(across(contains("conv"), abs)) %>%
          mutate(paper_id = select(., claim_id) %>% apply(1, function(x) str_split(x, "_") %>% unlist() %>% first())) %>%
          left_join(paper_metadata %>% select(paper_id, field = COS_pub_category), by = "paper_id")

        effects_combined <- merge(orig_effects,repli_effects,by="claim_id",all.x = TRUE,all.y = FALSE)

        effects_combined <- effects_combined %>%
          group_by(paper_id) %>%
          mutate(weight = 1/n()) %>%
          ungroup()

        table_3 <- do.call(rbind,lapply(fields.order, function(field) {
          # field <- fields.order[1]
          data <- effects_combined[effects_combined$field==field,]
          
          c1.numerator <- sum(as.numeric(data$success)*data$weight)
          c1.numerator.formatted <- format.round(c1.numerator,1)
          c1.denom <- length(unique(data$paper_id))
          c1.denom.formatted <- format.round(c1.denom,0)
          c1.p <- sum(as.numeric(data$success)*data$weight)/length(unique(data$paper_id))
          c1.p.formatted <- paste0(format.round(100*c1.p,1),"%")
          c1 <- paste0(c1.numerator.formatted," / ",c1.denom.formatted," (",
                       c1.p.formatted,")")
          
          c1 <- paste0(c1.numerator.formatted," / ",c1.denom.formatted)
          
          c2 <- paste0(c1.p.formatted)
          
          data_orig <- data %>%
            filter(!is.na(orig_conv_r))%>%
            group_by(paper_id) %>%
            mutate(weight = 1/n()) %>%
            ungroup()
          
          data_repli <- data %>%
            filter(!is.na(repli_conv_r))%>%
            group_by(paper_id) %>%
            mutate(weight = 1/n()) %>%
            ungroup()
          
          data_repli_orig <- data %>%
            filter(!is.na(orig_conv_r))%>%
            filter(!is.na(repli_conv_r))%>%
            group_by(paper_id) %>%
            mutate(weight = 1/n()) %>%
            ungroup()
          
          #c2 <- sum(data_repli_orig$weight)
          
          c3.median <- weighted.median(data_repli_orig$orig_conv_r,data_repli_orig$weight)
          c3.SD <- sqrt(wtd.var(data_repli_orig$orig_conv_r,data_repli_orig$weight))
          c3 <- paste0(format.round(c3.median,2)," (",format.round(c3.SD,2),")")
          
          c4.median <- weighted.median(data_repli_orig$repli_conv_r,data_repli_orig$weight)
          c4.SD <- sqrt(wtd.var(data_repli_orig$repli_conv_r,data_repli_orig$weight))
          c4 <- paste0(format.round(c4.median,2)," (",format.round(c4.SD,2),")")
          
          data.frame(field,c1,c2,c3,c4)
        }))
        
        table_3 <- table_3[c("c1","c2","c3","c4")]

        for (row in 1:nrow(table_3)){
          for (col in 1:ncol(table_3)){
            assign(paste0("table_3_",row,"_",col),
                   table_3[row,col])
          }
        }
        
        table_3_denoms <- as.numeric(sub(".* / ", "", table_3$c1))
        
        
        table_3_g20_repli_rate_max <- sub("%.*", "", table_3[table_3_denoms>20,]$c2) %>%
          as.numeric() %>% max()
        table_3_g20_repli_rate_min <- sub("%.*", "", table_3[table_3_denoms>20,]$c2) %>%
          as.numeric() %>% min()
        
        rm(orig_effects,repli_effects,effects_combined)
        
      }
      
      # Table 4
      {
        repli_effects <- repli_outcomes %>%
          filter(!is_covid) %>%
          filter(repli_version_of_record) %>%
          select(report_id, claim_id,repli_sample_size_value, success = repli_score_criteria_met, repli_type,
                 repli_conv_r, repli_conv_r_lb, repli_conv_r_ub, repli_effect_size_type, repli_effect_size_value,
                 rr_power_100_original_effect,rr_power_100_original_effect_design,
                 repli_power_for_50_effect,rr_power_50_original_effect_design,) %>%
          mutate(ser_effect = ifelse(str_detect(repli_effect_size_type, "ser_"), abs(repli_effect_size_value), NA)) %>%
          mutate(across(contains("conv"), abs))
        
        orig_effects <- orig_outcomes %>%
          semi_join(repli_effects, by = "claim_id") %>%
          select(claim_id, paper_id,orig_sample_size_value, orig_conv_r, orig_conv_r_lb, orig_conv_r_ub,
                 orig_effect_size_type_repli, orig_effect_size_value_repli,
                 orig_power_for_50_effect) %>%
          mutate(ser_effect = ifelse(str_detect(orig_effect_size_type_repli, "ser_"), abs(orig_effect_size_value_repli), NA)) %>%
          mutate(across(contains("conv"), abs)) %>%
          mutate(paper_id = select(., claim_id) %>% apply(1, function(x) str_split(x, "_") %>% unlist() %>% first())) %>%
          left_join(paper_metadata %>% select(paper_id, field = COS_pub_category), by = "paper_id")
        
        effects_combined <- merge(orig_effects,repli_effects,by="claim_id",all.x = TRUE,all.y = FALSE)
        
        effects_combined <- effects_combined %>%
          group_by(paper_id) %>%
          mutate(weight = 1/n()) %>%
          ungroup()
        
        r1.data <- effects_combined[effects_combined$repli_type=="new data",]
        r1.numerator <- c(length(unique(r1.data$paper_id)),sum(r1.data$success*r1.data$weight),nrow(r1.data),sum(r1.data$success))
        r1.numerator.formatted <- c(format.round(r1.numerator[1],0),format.round(r1.numerator[2],1),format.round(r1.numerator[3:4],0))
        r1.denom <- c(length(unique(r1.data$paper_id)),length(unique(r1.data$paper_id)),nrow(r1.data),nrow(r1.data))
        r1.denom.formatted <- format.round(r1.denom,0)
        r1.p <- r1.numerator/r1.denom
        r1.p.formatted <- paste0(format.round(100*r1.p,1),"%")
        r1 <- paste0(r1.numerator.formatted," / ",r1.denom.formatted,"\n(",r1.p.formatted,")")
        
        
        r2.data <- effects_combined[effects_combined$repli_type=="secondary data",]
        r2.numerator <- c(length(unique(r2.data$paper_id)),sum(r2.data$success*r2.data$weight),nrow(r2.data),sum(r2.data$success))
        r2.numerator.formatted <- c(format.round(r2.numerator[1],0),format.round(r2.numerator[2],1),format.round(r2.numerator[3:4],0))
        r2.denom <- c(length(unique(r2.data$paper_id)),length(unique(r2.data$paper_id)),nrow(r2.data),nrow(r2.data))
        r2.denom.formatted <- format.round(r2.denom,0)
        r2.p <- r2.numerator/r2.denom
        r2.p.formatted <- paste0(format.round(100*r2.p,1),"%")
        r2 <- paste0(r2.numerator.formatted," / ",r2.denom.formatted,"\n(",r2.p.formatted,")")
        
        r3.data <- r1.data %>%
          filter(!is.na(orig_conv_r)) %>%
          group_by(paper_id) %>%
          mutate(weight = 1/n()) %>%
          ungroup()
        r3.median <- c(weighted.quantile(r3.data$orig_conv_r,r3.data$weight,.5),
                       weighted.quantile(r3.data$repli_conv_r,r3.data$weight,.5),
                       quantile(r3.data$orig_conv_r,.5,na.rm=TRUE),
                       quantile(r3.data$repli_conv_r,.5,na.rm=TRUE))
        r3.median.formatted <- format.round(r3.median,2)
        r3.SD <- c(sqrt(wtd.var(r3.data$orig_conv_r,r3.data$weight)),
                   sqrt(wtd.var(r3.data$repli_conv_r,r3.data$weight)),
                   sd(r3.data$orig_conv_r,1),
                   sd(r3.data$repli_conv_r,1))
        r3.SD.formatted <- format.round(r3.SD,2)
        r3 <- paste0(r3.median.formatted," (",r3.SD.formatted,")")
        
        r4.data <- r2.data %>%
          filter(!is.na(orig_conv_r)) %>%
          group_by(paper_id) %>%
          mutate(weight = 1/n()) %>%
          ungroup()
        r4.median <- c(weighted.quantile(r4.data$orig_conv_r,r4.data$weight,.5),
                       weighted.quantile(r4.data$repli_conv_r,r4.data$weight,.5),
                       quantile(r4.data$orig_conv_r,.5,na.rm=TRUE),
                       quantile(r4.data$repli_conv_r,.5,na.rm=TRUE))
        r4.median.formatted <- format.round(r4.median,2)
        r4.SD <- c(sqrt(wtd.var(r4.data$orig_conv_r,r4.data$weight)),
                   sqrt(wtd.var(r4.data$repli_conv_r,r4.data$weight)),
                   sd(r4.data$orig_conv_r,1),
                   sd(r4.data$repli_conv_r,1))
        r4.SD.formatted <- format.round(r4.SD,2)
        r4 <- paste0(r4.median.formatted," (",r4.SD.formatted,")")
        
        table_4 <- rbind(r1,r2,r3,r4)
        table_4_numerator <- rbind(r1.numerator,r2.numerator,rep(NA,ncol(table_4)),rep(NA,ncol(table_4)))
        table_4_numerator_formatted <- rbind(r1.numerator.formatted,r2.numerator.formatted,rep(NA,ncol(table_4)),rep(NA,ncol(table_4)))
        table_4_denom <- rbind(r1.denom,r2.denom,rep(NA,ncol(table_4)),rep(NA,ncol(table_4)))
        table_4_denom_formatted <- rbind(r1.denom.formatted,r2.denom.formatted,rep(NA,ncol(table_4)),rep(NA,ncol(table_4)))
        table_4_p <- rbind(r1.p,r2.p,rep(NA,ncol(table_4)),rep(NA,ncol(table_4)))
        table_4_p_formatted <- rbind(r1.p.formatted,r2.p.formatted,rep(NA,ncol(table_4)),rep(NA,ncol(table_4)))
        
        table_4_median <- rbind(rep(NA,ncol(table_4)),rep(NA,ncol(table_4)),r3.median,r4.median)
        table_4_median_formatted <- rbind(rep(NA,ncol(table_4)),rep(NA,ncol(table_4)),r3.median.formatted,r4.median.formatted)
        table_4_SD <- rbind(rep(NA,ncol(table_4)),rep(NA,ncol(table_4)),r3.SD,r4.SD)
        table_4_SD_formatted <- rbind(rep(NA,ncol(table_4)),rep(NA,ncol(table_4)),r3.SD.formatted,r4.SD.formatted)
        
        for (row in 1:nrow(table_4)){
          for (col in 1:ncol(table_4)){
            assign(paste0("table_4_",row,"_",col),
                   table_4[row,col])
            
            assign(paste0("table_4_numerator_",row,"_",col),
                   table_4_numerator[row,col])
            assign(paste0("table_4_numerator_formatted_",row,"_",col),
                   table_4_numerator_formatted[row,col])
            
            assign(paste0("table_4_denom_",row,"_",col),
                   table_4_denom[row,col])
            assign(paste0("table_4_denom_formatted_",row,"_",col),
                   table_4_denom_formatted[row,col])
            
            assign(paste0("table_4_p_",row,"_",col),
                   table_4_p[row,col])
            assign(paste0("table_4_p_formatted_",row,"_",col),
                   table_4_p_formatted[row,col])
            
            assign(paste0("table_4_median_",row,"_",col),
                   table_4_median[row,col])
            assign(paste0("table_4_median_formatted_",row,"_",col),
                   table_4_median_formatted[row,col])
            
            assign(paste0("table_4_SD_",row,"_",col),
                   table_4_SD[row,col])
            assign(paste0("table_4_SD_formatted_",row,"_",col),
                   table_4_SD_formatted[row,col])
          }
        }
        rm(r1,r2,r3,r4)
        
        # Add rows 5 and 6
        
        table_4_5_13.data <- repli_outcomes %>%
          filter(!is.na(repli_power_for_50_effect),
                 repli_type=="new data") %>%
          group_by(paper_id) %>%
          mutate(weight = 1/n()) %>%
          ungroup()
        
        table_4_5_1 <- format.round(weighted.median(table_4_5_13.data$repli_power_for_50_effect,table_4_5_13.data$weight),2)
        table_4_5_3 <- format.round(median(table_4_5_13.data$repli_power_for_50_effect),2)
        
        table_4_5_24.data <- repli_outcomes %>%
          filter(!is.na(rr_power_50_original_effect_design),
                 repli_type=="new data") %>%
          group_by(paper_id) %>%
          mutate(weight = 1/n()) %>%
          ungroup()
        
        table_4_5_2 <- format.round(weighted.median(table_4_5_24.data$rr_power_50_original_effect_design,table_4_5_24.data$weight),2)
        table_4_5_4 <- format.round(median(table_4_5_24.data$rr_power_50_original_effect_design),2)
        
        table_4_6_13.data <- repli_outcomes %>%
          filter(!is.na(repli_power_for_50_effect),
                 repli_type=="secondary data") %>%
          group_by(paper_id) %>%
          mutate(weight = 1/n()) %>%
          ungroup()
        
        table_4_6_1 <- format.round(weighted.median(table_4_6_13.data$repli_power_for_50_effect,table_4_6_13.data$weight),2)
        table_4_6_3 <- format.round(median(table_4_6_13.data$repli_power_for_50_effect),2)
        
        table_4_6_24.data <- repli_outcomes %>%
          filter(!is.na(rr_power_50_original_effect_design),
                 repli_type=="secondary data") %>%
          group_by(paper_id) %>%
          mutate(weight = 1/n()) %>%
          ungroup()
        
        table_4_6_2 <- format.round(weighted.median(table_4_6_24.data$rr_power_50_original_effect_design,table_4_6_24.data$weight),2)
        table_4_6_4 <- format.round(median(table_4_6_24.data$rr_power_50_original_effect_design),2)
      }
      
      # Table 5
      {
        table_5_1_1 <- repli_outcomes %>%
          filter(repli_type == "new data") %>%
          dplyr::summarize(med_power_75 = 100*weighted.median(combined_power_75, weight, na.rm = TRUE)) %>%
          pull(med_power_75) %>%
          format.round(1) %>% paste0("%")
        
        table_5_2_1 <- repli_outcomes %>%
          filter(repli_type == "secondary data") %>%
          dplyr::summarize(med_power_75 = 100*weighted.median(combined_power_75, weight, na.rm = TRUE)) %>%
          pull(med_power_75) %>%
          format.round(1) %>% paste0("%")
        
        table_5_1_2 <- repli_outcomes %>%
          filter(repli_type == "new data") %>%
          dplyr::summarize(med_power_75 = 100*median(combined_power_75, na.rm = T)) %>%
          pull(med_power_75) %>%
          format.round(1) %>% paste0("%")
        
        table_5_2_2 <- repli_outcomes %>%
          filter(repli_type == "secondary data") %>%
          dplyr::summarize(med_power_75 = 100*median(combined_power_75, na.rm = T)) %>%
          pull(med_power_75) %>%
          format.round(1) %>% paste0("%")
        
        table_5 <- as.data.frame(matrix(c(table_5_1_1,table_5_2_1,table_5_1_2,table_5_2_2),nrow=2))
      }

    }
    
    # Introduction
    {
      n_journals_business <- length(unique(
        repli_outcomes_merged[repli_outcomes_merged$field=="Business",]$publication_standard
      ))
      n_journals_econ <- length(unique(
        repli_outcomes_merged[repli_outcomes_merged$field=="Economics and Finance",]$publication_standard
      ))
      n_journals_edu <- length(unique(
        repli_outcomes_merged[repli_outcomes_merged$field=="Education",]$publication_standard
      ))
      n_journals_polisci <- length(unique(
        repli_outcomes_merged[repli_outcomes_merged$field=="Political Science",]$publication_standard
      ))
      n_journals_psych <- length(unique(
        repli_outcomes_merged[repli_outcomes_merged$field=="Psychology and Health",]$publication_standard
      ))
      n_journals_soc <- length(unique(
        repli_outcomes_merged[repli_outcomes_merged$field=="Sociology and Criminology",]$publication_standard
      ))
                                           
    }
    
    # Results: Replications completed by discipline and year in comparison with the sampling frame
    {
      p_rep_edu_initial <- 
        paste0(format.round(100*table_1_n$Education[1]/table_1_n$Total[1],1),"% [n = ",table_1_n$Education[1],"/",table_1_n$Total[1],"]")
      p_rep_edu_replis <- 
        paste0(format.round(100*table_1_n$Education[6]/table_1_n$Total[6],1),"% [n = ",table_1_n$Education[6],"/",table_1_n$Total[6],"]")
      
      p_rep_polisci_initial <- 
        paste0(format.round(100*table_1_n$`Political Science`[1]/table_1_n$Total[1],1),"% [n = ",table_1_n$`Political Science`[1],"/",table_1_n$Total[1],"]")
      p_rep_polisci_replis <- 
        paste0(format.round(100*table_1_n$`Political Science`[6]/table_1_n$Total[6],1),"% [n = ",table_1_n$`Political Science`[6],"/",table_1_n$Total[6],"]")
      
      p_rep_psych_initial <- 
        paste0(format.round(100*table_1_n$`Psychology and Health`[1]/table_1_n$Total[1],1),"% [n = ",table_1_n$`Psychology and Health`[1],"/",table_1_n$Total[1],"]")
      p_rep_psych_replis <- 
        paste0(format.round(100*table_1_n$`Psychology and Health`[6]/table_1_n$Total[6],1),"% [n = ",table_1_n$`Psychology and Health`[6],"/",table_1_n$Total[6],"]")
      p_rep_psych_final <- 
        paste0(format.round(100*table_1_n$`Psychology and Health`[8]/table_1_n$Total[8],1),"% [n = ",table_1_n$`Psychology and Health`[8],"/",table_1_n$Total[8],"]")
      
      p_rep_business_initial <- 
        paste0(format.round(100*table_1_n$Business[1]/table_1_n$Total[1],1),"% [n = ",table_1_n$Business[1],"/",table_1_n$Total[1],"]")
      p_rep_business_final <- 
        paste0(format.round(100*table_1_n$Business[8]/table_1_n$Total[8],1),"% [n = ",table_1_n$Business[8],"/",table_1_n$Total[8],"]")
      
      p_rep_min_diff_econ_edu_polisci_soc <- paste0(format.round(100*max(abs(table_1_p[c("Economics and Finance","Education","Political Science","Sociology and Criminology")][1,]-
        table_1_p[c("Economics and Finance","Education","Political Science","Sociology and Criminology")][8,])),0),"%")
      
      
      
      n_papers_init_sample_p1 <- nrow(paper_status_tracking %>% filter(p1_delivery))
      
      n_papers_p1_repli_completed <- length(unique(repli_outcomes[repli_outcomes$type_internal!="p2",]$paper_id))
      n_papers_p2_repli_completed <- length(unique(repli_outcomes[repli_outcomes$type_internal=="p2",]$paper_id))
      
      n_papers_eligible_repli_p1 <- nrow(paper_status_tracking %>% filter(RR))
      
      n_papers_eligible_repli_p1_multi_claim <- nrow(paper_status_tracking %>% filter(bushel))
      
      n_papers_eligible_repli_p1_single_claim <- n_papers_eligible_repli_p1-n_papers_eligible_repli_p1_multi_claim
      
      n_papers_init_sample_p2 <- nrow(paper_status_tracking %>% filter(p2_delivery))
      
      n_papers_repli_p2 <- nrow(repli_outcomes %>% filter(type_internal=="p2"))
      
      repli_binary_score_p1 <-
        na.omit(merge(repli_binary[c("paper_id","claim_id","repli_score_criteria_met")],
              repli_outcomes[c("claim_id","type_internal")],
              by="claim_id",all.x=TRUE,all.y=FALSE)) %>%
        filter(type_internal!="p2") %>% 
        group_by(paper_id) %>% 
        mutate(weight = 1/n()) %>%
        ungroup()
      
      p_repli_success_score_criteria_met_p1 <- paste0(format.round(
        100*sum(repli_binary_score_p1$repli_score_criteria_met*repli_binary_score_p1$weight)/
        sum(repli_binary_score_p1$weight),1),"%")
      
      repli_binary_score_p2 <-
        na.omit(merge(repli_binary[c("paper_id","claim_id","repli_score_criteria_met")],
                      repli_outcomes[c("claim_id","type_internal")],
                      by="claim_id",all.x=TRUE,all.y=FALSE)) %>%
        filter(type_internal=="p2") %>% 
        group_by(paper_id) %>% 
        mutate(weight = 1/n()) %>%
        ungroup()
      
      p_repli_success_score_criteria_met_p2 <- paste0(format.round(
        100*sum(repli_binary_score_p2$repli_score_criteria_met*repli_binary_score_p2$weight)/
          sum(repli_binary_score_p2$weight),1),"%")
    }
    
    # Results: General
    {
      n_claims <- length(unique(repli_outcomes_merged$claim_id))
      
      n_papers <- length(unique(repli_outcomes_merged$paper_id))
      
      n_journals <- length(unique(repli_outcomes_merged$publication_standard))
    }
    
    # Results: Evaluating the replication outcome against the null hypothesis of no effect
    {
      n_papers_repli_stat_sig_claims_same_dir <- 
        format.round(sum(as.numeric((!is.na(repli_outcomes$repli_p_value) & repli_outcomes$repli_p_value <= 0.05 & 
                          !is.na(repli_outcomes$repli_pattern_criteria_met) & repli_outcomes$repli_pattern_criteria_met==TRUE))*
            repli_outcomes$weight
        ),1)
      
      p_papers_repli_stat_sig_claims_same_dir <- 
        bootstrap.clust(data=repli_outcomes,
                        FUN=function(x) {
                          x <- x %>% group_by(paper_id) %>% mutate(weight = 1/n())
                          weighted.mean((!is.na(x$repli_p_value) & x$repli_p_value <= 0.05 &
                                           !is.na(x$repli_pattern_criteria_met) & x$repli_pattern_criteria_met==TRUE),
                                        x$weight)
                        },
                        clustervar = "paper_id",
                        keepvars=c("repli_p_value","repli_pattern_criteria_met"),
                        alpha=.05,tails="two-tailed",iters=iters,
                        format.percent=TRUE,digits=1,na.rm = FALSE
        )$formatted.text
      
      
      n_papers_repli_stat_sig_claims_opp_dir <- 
        format.round(sum(as.numeric((!is.na(repli_outcomes$repli_p_value) & repli_outcomes$repli_p_value <= 0.05 & 
                          !is.na(repli_outcomes$repli_pattern_criteria_met) & repli_outcomes$repli_pattern_criteria_met==FALSE))*
              repli_outcomes$weight
        ),1)
      
      p_papers_repli_stat_sig_claims_opp_dir <- 
        bootstrap.clust(data=repli_outcomes,
                        FUN=function(x) {
                          x <- x %>% group_by(paper_id) %>% mutate(weight = 1/n())
                          weighted.mean((!is.na(x$repli_p_value) & x$repli_p_value <= 0.05 &
                                           !is.na(x$repli_pattern_criteria_met) & x$repli_pattern_criteria_met==FALSE),
                                        x$weight)
                        },
                        clustervar = "paper_id",
                        keepvars=c("repli_p_value","repli_pattern_criteria_met"),
                        alpha=.05,tails="two-tailed",iters=iters,
                        format.percent=TRUE,digits=1,na.rm = FALSE
        )$formatted.text
      
      n_papers_repli_non_stat_sig <- 
        format.round(sum(as.numeric(!is.na(repli_outcomes$repli_p_value) & repli_outcomes$repli_p_value > 0.05)*
              repli_outcomes$weight),1)
      
      p_papers_repli_non_stat_sig <- 
        bootstrap.clust(data=repli_outcomes,
                        FUN=function(x) {
                          x <- x %>% group_by(paper_id) %>% mutate(weight = 1/n())
                          weighted.mean(!is.na(x$repli_p_value) & as.numeric(x$repli_p_value > 0.05),
                                        x$weight)
                        },
                        clustervar = "paper_id",
                        keepvars=c("repli_p_value","repli_pattern_criteria_met"),
                        alpha=.05,tails="two-tailed",iters=iters,
                        format.percent=TRUE,digits=1,na.rm = FALSE
        )$formatted.text
      
      n_claims_repli_stat_sig_claims_same_dir <- 
        sum(as.numeric((!is.na(repli_outcomes$repli_p_value) & repli_outcomes$repli_p_value <= 0.05 & 
                          !is.na(repli_outcomes$repli_pattern_criteria_met) & repli_outcomes$repli_pattern_criteria_met==TRUE))
        )
      
      p_claims_repli_stat_sig_claims_same_dir <- 
        format.text.percent(sum(as.numeric((!is.na(repli_outcomes$repli_p_value) & repli_outcomes$repli_p_value <= 0.05 & 
                                            !is.na(repli_outcomes$repli_pattern_criteria_met) & repli_outcomes$repli_pattern_criteria_met==TRUE))),
                              n_claims)
      
      n_claims_repli_stat_sig_claims_opp_dir <- 
        sum(as.numeric((!is.na(repli_outcomes$repli_p_value) & repli_outcomes$repli_p_value <= 0.05 & 
                          !is.na(repli_outcomes$repli_pattern_criteria_met) & repli_outcomes$repli_pattern_criteria_met==FALSE))
        )
      
      p_claims_repli_stat_sig_claims_opp_dir <- 
        format.text.percent(sum(as.numeric((!is.na(repli_outcomes$repli_p_value) & repli_outcomes$repli_p_value <= 0.05 & 
                                              !is.na(repli_outcomes$repli_pattern_criteria_met) & repli_outcomes$repli_pattern_criteria_met==FALSE))),
                            n_claims)
      
      n_claims_repli_non_stat_sig <- 
        sum(as.numeric(!is.na(repli_outcomes$repli_p_value) & repli_outcomes$repli_p_value > 0.05))
      
      p_claims_repli_non_stat_sig <- 
        format.text.percent(sum(as.numeric(!is.na(repli_outcomes$repli_p_value) & repli_outcomes$repli_p_value > 0.05)),
                            n_claims)
      
      df.power <- repli_outcomes %>%
        filter(!is.na(rr_power_100_original_effect)) %>% group_by(paper_id) %>% 
        mutate(weight = 1/n()) %>%
        ungroup()
      
      repli_outcomes %>%
        filter(!is_covid & repli_version_of_record) %>%
        mutate(combined_power_100 = 100*combined_power_100) %>%
        pull(combined_power_100) %>%
        median(na.rm = T) %>%
        round(1)
      
      median_power_100_effect <- repli_outcomes %>%
        mutate(combined_power_100 = 100*combined_power_100) %>%
        pull(combined_power_100) %>%
        median(na.rm = T) %>%
        format.round(1) %>%
        paste0("%")
      
      median_power_100_effect_secondary_data <- repli_outcomes %>%
        filter(repli_type == "secondary data") %>%
        mutate(combined_power_100 = 100*combined_power_100) %>%
        pull(combined_power_100) %>%
        median(na.rm = T) %>%
        format.round(1) %>%
        paste0("%")
      
      median_power_100_effect_new_data <- repli_outcomes %>%
        filter(repli_type == "new data") %>%
        mutate(combined_power_100 = 100*combined_power_100) %>%
        pull(combined_power_100) %>%
        median(na.rm = T) %>%
        format.round(1) %>%
        paste0("%")
      
      mean_power_100_effect_secondary_data <- repli_outcomes %>%
        filter(repli_type == "secondary data") %>%
        mutate(combined_power_100 = 100*combined_power_100) %>%
        pull(combined_power_100) %>%
        mean(na.rm = T) %>%
        format.round(1) %>%
        paste0("%")
      
      mean_power_100_effect_new_data <- repli_outcomes %>%
        filter(repli_type == "new data") %>%
        mutate(combined_power_100 = 100*combined_power_100) %>%
        pull(combined_power_100) %>%
        mean(na.rm = T) %>%
        format.round(1) %>%
        paste0("%")
      
      median_power_50_effect <- repli_outcomes %>%
        mutate(combined_power_100 = 100*combined_power_100) %>%
        dplyr::summarize(prop_50 = 100*mean(combined_power_100 >= 50, na.rm = T)) %>%
        pull(prop_50) %>%
        format.round(1) %>%
        paste0("%")
      
      median_power_75_effect <- repli_outcomes %>%
        mutate(combined_power_100 = 100*combined_power_100) %>%
        dplyr::summarize(prop_75 = 100*mean(combined_power_100 >= 75, na.rm = T)) %>%
        pull(prop_75) %>%
        format.round(1) %>%
        paste0("%")
      
      p_repli_50_power_for_100_effect <- 
        paste0(format.round(
          100*
            sum(as.integer(df.power$rr_power_100_original_effect >=.5) * df.power$weight)/
            sum( df.power$weight),
          digits=1),"%")
      
      p_repli_75_power_for_100_effect <- 
        paste0(format.round(
          100*
            sum(as.integer(df.power$rr_power_100_original_effect >=.75) * df.power$weight)/
            sum( df.power$weight),
          digits=1),"%")
      
      df.power <- repli_outcomes[!is.na(repli_outcomes$repli_power_for_50_effect),] %>%
        group_by(paper_id) %>%
        mutate(weight = 1/n())
      
      p_repli_90_power_for_50_effect <- 
        paste0(format.round(
          100*
            sum(as.integer(df.power$repli_power_for_50_effect >=.9) * df.power$weight)/
            sum( df.power$weight),
          digits=1),"%")
      
      df.power <- repli_outcomes[!is.na(repli_outcomes$repli_power_for_50_effect),] %>%
        group_by(paper_id) %>%
        mutate(weight = 1/n())
      
      p_repli_90_power_for_50_effect <- 
        paste0(format.round(
          100*
            sum(as.integer(df.power$repli_power_for_50_effect >=.9) * df.power$weight)/
            sum( df.power$weight),
          digits=1),"%")
      
      df.power <- repli_outcomes %>%
        filter(!is.na(rr_power_100_original_effect_design)) %>% group_by(paper_id) %>% 
        mutate(weight = 1/n()) %>%
        ungroup()
      
      median_power_100_effect_design <- paste0(format.round(
        100*weighted.median(df.power$rr_power_100_original_effect_design,df.power$weight),
        digits=1),"%")
      
      p_repli_50_power_for_100_effect_design <- 
        paste0(format.round(
          100*
            sum(as.integer(df.power$rr_power_100_original_effect_design >=.5) * df.power$weight)/
            sum( df.power$weight),
          digits=1),"%")
      
      p_repli_75_power_for_100_effect_design <- 
        paste0(format.round(
          100*
            sum(as.integer(df.power$rr_power_100_original_effect_design >=.75) * df.power$weight)/
            sum( df.power$weight),
          digits=1),"%")
      
      
      rm(df.power)
      
      n_bushel_claims_selected <- nrow(claims_non_significant_bushels)
      
      n_bushel_claims_selected_sig <- claims_non_significant_bushels %>%
        filter(nonsig != "T") %>% nrow()
      
      p_bushel_claims_selected_sig <- paste0(format.round(
        100*n_bushel_claims_selected_sig/n_bushel_claims_selected,1),"%")
      
      claims_non_significant_bushels %>%
        count(nonsig != "T") %>%
        mutate(prop = n/sum(n)) %>%
        filter(`nonsig != "T"`) %>%
        pull(prop)

    }
    
    # Results: Evaluating replication success with a variety of binary assessments
    {
      # Prepare binary assessment data
      {
        
        binary.proportions <- 
          do.call(rbind,lapply(df.binvars$binvars.raw,function(binary.var){
            repli_binary_temp <- na.omit(repli_binary[c("paper_id",binary.var)])
            repli_binary_temp <- repli_binary_temp %>%
              group_by(paper_id) %>%
              mutate(weight = 1/n())
            
            n_testable_claims <- nrow(repli_binary_temp)
            p_testable_claims <- n_testable_claims/length(unique(repli_outcomes$claim_id))
            n_passed_claims <- sum(repli_binary_temp[[binary.var]])
            p_passed_temp <- binconf(n_passed_claims,n_testable_claims)
            p_passed_claims <- p_passed_temp[1]
            p_passed_CI_LB_claims <- p_passed_temp[2]
            p_passed_CI_UB_claims <- p_passed_temp[3]
            
            
            n_testable_papers <- length(unique((repli_binary_temp$paper_id)))
            p_testable_papers <- n_testable_papers/length(unique(repli_outcomes$paper_id))
            n_passed_papers <- sum(repli_binary_temp[[binary.var]]*repli_binary_temp$weight)
            p_passed_temp <- cw.proportion(as.integer(repli_binary_temp[[binary.var]]),
                                              repli_binary_temp$weight,repli_binary_temp$paper_id,iters)
            p_passed_papers <- p_passed_temp$point.estimate
            p_passed_CI_LB_papers <- p_passed_temp$CI.lb
            p_passed_CI_UB_papers <- p_passed_temp$CI.ub
            
            data.frame(binary.var,
                       n_testable_claims,p_testable_claims,n_passed_claims,p_passed_claims,p_passed_CI_LB_claims,p_passed_CI_UB_claims,
                       n_testable_papers,p_testable_papers,n_passed_papers,p_passed_papers,p_passed_CI_LB_papers,p_passed_CI_UB_papers
                       )
          }))
        
      }
      
      n_binary_measures <- nrow(df.binvars)
      
      p_claims_binary_measures_measurable_min <- paste0(format.round(min(binary.proportions$p_testable_claims)*100,1),"%")
      p_claims_binary_measures_measurable_max <- paste0(format.round(max(binary.proportions$p_testable_claims)*100,1),"%")
      
      p_papers_binary_measures_measurable_min <- paste0(format.round(min(binary.proportions$p_testable_papers)*100,1),"%")
      p_papers_binary_measures_measurable_max <- paste0(format.round(max(binary.proportions$p_testable_papers)*100,1),"%")
    
      p_binary_passed_papers_min <- paste0(format.round(min(binary.proportions$p_passed_papers)*100,1),"%")
      p_binary_passed_papers_max <- paste0(format.round(max(binary.proportions$p_passed_papers)*100,1),"%")
      p_binary_passed_papers_median <- paste0(format.round(median(binary.proportions$p_passed_papers)*100,1),"%")
      
      p_binary_passed_claims_min <- paste0(format.round(min(binary.proportions$p_passed_claims)*100,1),"%")
      p_binary_passed_claims_max <- paste0(format.round(max(binary.proportions$p_passed_claims)*100,1),"%")
      
      p_binary_passed_papers_MA <- paste0(format.round(binary.proportions[binary.proportions$binary.var=="repli_binary_meta_success",]$p_passed_papers*100,1),"%")
      
      p_binary_passed_papers_orig_wthn <- paste0(format.round(binary.proportions[binary.proportions$binary.var=="repli_binary_orig_wthn",]$p_passed_papers*100,1),"%")
      p_binary_passed_papers_rep_wthn <- paste0(format.round(binary.proportions[binary.proportions$binary.var=="repli_binary_rep_wthn",]$p_passed_papers*100,1),"%")
    
      n_all_measures_claims <- nrow(na.omit(repli_binary))
      
      
      # Generate correlations matrix
      {
        # repli_binary <- repli_outcomes[c("paper_id", "claim_id",
        #                                  df.binvars$binvars.raw)]
        
        # Summarize by proportions w/ CIs
        repli_binary_binaries <- repli_binary[,3:ncol(repli_binary)]
        repli_binary_binaries <- repli_binary_binaries[c(binvars.order)]
        colnames(repli_binary_binaries) <- df.binvars$binvars.full.text
        
        # Generate a results matrix
        repli_binary_cors <- matrix(NA,nrow=ncol(repli_binary_binaries),ncol=ncol(repli_binary_binaries))
        rownames(repli_binary_cors) <- colnames (repli_binary_cors) <- colnames(repli_binary_binaries)
        
        get_wt_corr <- function(i.x,i.y){
          df.cors <- repli_binary_binaries[,c(i.x,i.y)]
          df.cors$paper_id <- repli_binary$paper_id
          df.cors <- na.omit(df.cors)
          df.cors <- df.cors %>%
            group_by(paper_id) %>%
            mutate(weight = 1/n())
          
          weightedCorr(as.integer(unlist(df.cors[,1])),as.integer(unlist(df.cors[,2])),weights=df.cors$weight,method="Spearman")
        }
        
        # Get pairwise complete weighted correlations
        invisible(lapply(1:nrow(repli_binary_cors),function(i){
          lapply(1:(i),function(j){
            if(i!=j){
              repli_binary_cors[i,j] <<- get_wt_corr(i,j)
            }
          })
        }))
      }
      cors.list <- c(na.omit(c(repli_binary_cors)))
      
      repli_binary_cors_intervals <- repli_binary_cors[c(df.binvars[df.binvars$binvars.group=="Contained within interval measures",]$binvars.full.text),c(df.binvars[df.binvars$binvars.group=="Contained within interval measures",]$binvars.full.text)]
      cors_list_intervals <-  paste(format.round(sort(c(na.omit(c(repli_binary_cors_intervals)))),2),collapse=", ")
      
      repli_binary_cors_intervals_vs_non_interval <- c(c(na.omit(c(repli_binary_cors[c(df.binvars[df.binvars$binvars.group=="Contained within interval measures",]$binvars.full.text),c(df.binvars[df.binvars$binvars.group!="Contained within interval measures",]$binvars.full.text)]))),
                                                       c(na.omit(c(repli_binary_cors[c(df.binvars[df.binvars$binvars.group!="Contained within interval measures",]$binvars.full.text),c(df.binvars[df.binvars$binvars.group=="Contained within interval measures",]$binvars.full.text)]))))
      repli_binary_cors_intervals_vs_non_interval_median <- format.round(median(repli_binary_cors_intervals_vs_non_interval),2)
      repli_binary_cors_intervals_vs_non_interval_min <- format.round(min(repli_binary_cors_intervals_vs_non_interval),2)
      repli_binary_cors_intervals_vs_non_interval_max <- format.round(max(repli_binary_cors_intervals_vs_non_interval),2)
      
      binary_cors_median <- format.round(median(cors.list),2)
      binary_cors_min <- format.round(min(cors.list),2)
      binary_cors_max <- format.round(max(cors.list),2)
      
      df.repli_binary_calculatable <- repli_binary %>%
        group_by(paper_id) %>%
        mutate(weight = 1/n())
      
      weights <- df.repli_binary_calculatable$weight
      lapply(3:(ncol(df.repli_binary_calculatable)-1),function(i) {
        df.repli_binary_calculatable[,i] <<- as.integer(!is.na(df.repli_binary_calculatable[,i])) * df.repli_binary_calculatable$weight
      })

      repli_binary_calculatable <- colSums(df.repli_binary_calculatable[,3:(ncol(df.repli_binary_calculatable)-1)])/sum(df.repli_binary_calculatable$weight)
      
      repli_binary_calculatable_min <- paste0(format.round(100*min(repli_binary_calculatable),1),"%")
      repli_binary_calculatable_max <- paste0(format.round(100*max(repli_binary_calculatable),1),"%")
      repli_binary_calculatable_median <- paste0(format.round(100*median(repli_binary_calculatable),1),"%")
    }
    
    # Results: Evaluating replication effect size against original effect size
    {
      
      all_effects <- repli_outcomes %>% 
        filter(!is_covid) %>% 
        filter(repli_version_of_record) %>% 
        select(paper_id,report_id, claim_id, repli_pattern_criteria_met, success = repli_score_criteria_met,
               repli_conv_r, repli_conv_r_lb, repli_conv_r_ub) %>% 
        left_join(
          orig_outcomes %>% 
            select(claim_id, orig_conv_r, orig_conv_r_lb, orig_conv_r_ub) %>% 
            distinct(),
          by = "claim_id"
        ) %>% 
        drop_na(repli_conv_r) %>% 
        drop_na(orig_conv_r) %>% 
        mutate(across(contains("orig"), abs)) %>% 
        mutate(
          across(
            contains("repli"),
            function(x) ifelse(repli_pattern_criteria_met, abs(x), -1*abs(x))
          )
        ) %>%
        group_by(paper_id) %>%
        mutate(weight = 1/n()) %>%
        ungroup()
      
      abs_effect_smaller_n_claims <- sum(abs(all_effects$repli_conv_r) < abs(all_effects$orig_conv_r))
      abs_effect_smaller_denom_claims <- sum(!is.na(all_effects$repli_conv_r) & !is.na(all_effects$orig_conv_r))
      abs_effect_smaller_p_claims <- paste0(format.round(100*abs_effect_smaller_n_claims/abs_effect_smaller_denom_claims,1),"%")
      
      abs_effect_smaller_n_papers <- sum((abs(all_effects$repli_conv_r) < abs(all_effects$orig_conv_r))*all_effects$weight)
      abs_effect_smaller_n_papers_formatted <- format.round(abs_effect_smaller_n_papers,1)
      abs_effect_smaller_denom_papers <- sum(all_effects$weight)
      abs_effect_smaller_p_papers <- paste0(format.round(100*abs_effect_smaller_n_papers/abs_effect_smaller_denom_papers,1),"%")
      
      # table_4_median_R_papers_neg_ratio <- 
      #   paste0(format.round(100*(table_4_median_4_1-table_4_median_4_2)/table_4_median_4_1,1),"%")
      # 
      # table_4_median_R_claims_neg_ratio <- 
      #   paste0(format.round(100*(table_4_median_4_3-table_4_median_4_4)/table_4_median_4_3,1),"%")
      
      corr_pearsons_orig_v_repli <- 
        format.round(weightedCorr(abs(all_effects$orig_conv_r),abs(all_effects$repli_conv_r),method="Spearman",
                   weights=all_effects$weight),2)
    }
    
    # Results: Original and replication effects by discipline
    {
      
      fields.order <- c("business","economics and finance","education","political science",
                        "psychology and health","sociology and criminology")
      
      repli_effects <- repli_outcomes %>%
        filter(!is_covid) %>%
        filter(repli_version_of_record) %>%
        select(report_id, claim_id,repli_sample_size_value, success = repli_score_criteria_met, repli_type,
               repli_conv_r, repli_conv_r_lb, repli_conv_r_ub, repli_effect_size_type, repli_effect_size_value) %>%
        mutate(ser_effect = ifelse(str_detect(repli_effect_size_type, "ser_"), abs(repli_effect_size_value), NA)) %>%
        mutate(across(contains("conv"), abs))
      
      orig_effects <- orig_outcomes %>%
        semi_join(repli_effects, by = "claim_id") %>%
        select(claim_id, paper_id,orig_sample_size_value, orig_conv_r, orig_conv_r_lb, orig_conv_r_ub,
               orig_effect_size_type_repli, orig_effect_size_value_repli) %>%
        mutate(ser_effect = ifelse(str_detect(orig_effect_size_type_repli, "ser_"), abs(orig_effect_size_value_repli), NA)) %>%
        mutate(across(contains("conv"), abs)) %>%
        mutate(paper_id = select(., claim_id) %>% apply(1, function(x) str_split(x, "_") %>% unlist() %>% first())) %>%
        left_join(paper_metadata %>% select(paper_id, field = COS_pub_category), by = "paper_id")
      
      effects_combined <- merge(orig_effects,repli_effects,by="claim_id",all.x = TRUE,all.y = FALSE)
      
      effects_combined <- effects_combined %>%
        group_by(paper_id) %>%
        mutate(weight = 1/n()) %>%
        ungroup()
      
      effects_by_field <- do.call(rbind,lapply(fields.order, function(field) {
        data <- effects_combined[effects_combined$field==field,]
        c1.numerator <- sum(as.numeric(data$success)*data$weight)
        c1.denom <- length(unique(data$paper_id))
        c1.p <- sum(as.numeric(data$success)*data$weight)/length(unique(data$paper_id))
        c2.numerator <- sum(as.numeric(data$success))
        c2.denom <- length(unique(data$claim_id))
        c2.p <- sum(as.numeric(data$success))/length(unique(data$claim_id))
        data.frame(field,
                   c1.numerator,
                   c1.denom,
                   c1.p,
                   c2.numerator,
                   c2.denom,
                   c2.p)
      }))
      
      # table_5_min_p_c1 <- paste0(format.round(min(100*table_5_p[,1]),1),"%")
      # table_5_max_p_c1 <- paste0(format.round(max(100*table_5_p[,1]),1),"%")
      # table_5_median_p_c1 <- paste0(format.round(median(100*table_5_p[,1]),1),"%")
      # table_5_min_p_c2 <- paste0(format.round(min(100*table_5_p[,2]),1),"%")
      # table_5_max_p_c2 <- paste0(format.round(max(100*table_5_p[,2]),1),"%")
      # table_5_median_p_c2 <- paste0(format.round(median(100*table_5_p[,2]),1),"%")
      
      repli_rate_field_min_p_papers <- paste0(format.round(min(100*effects_by_field$c1.p),1),"%")
      repli_rate_field_max_p_papers <- paste0(format.round(max(100*effects_by_field$c1.p),1),"%")
      repli_rate_field_median_p_papers <- paste0(format.round(median(100*effects_by_field$c1.p),1),"%")
      repli_rate_field_min_p_claims <- paste0(format.round(min(100*effects_by_field$c2.p),1),"%")
      repli_rate_field_max_p_claims <- paste0(format.round(max(100*effects_by_field$c2.p),1),"%")
      repli_rate_field_median_p_claims <- paste0(format.round(median(100*effects_by_field$c2.p),1),"%")
      
      chisq.fields <- wtd.chi.sq(effects_combined$success,effects_combined$field,weight=effects_combined$weight)
      chisq_success_fields_pvalue_papers <- format.round(chisq.fields[3],2)
      
      chisq.fields <- wtd.chi.sq(effects_combined$success,effects_combined$field,weight=rep(1,nrow(effects_combined)))
      chisq_success_fields_pvalue_claims <- format.round(chisq.fields[3],2)
    }
    
    # Results: Original and replication effects by year of original publication
    {
      repli_effects <- repli_outcomes %>% 
        filter(!is_covid) %>% 
        filter(repli_version_of_record) %>% 
        select(report_id, claim_id, repli_sample_size_value, success = repli_score_criteria_met, repli_type,
               repli_conv_r, repli_conv_r_lb, repli_conv_r_ub, repli_effect_size_type, repli_effect_size_value) %>%
        mutate(ser_effect = ifelse(str_detect(repli_effect_size_type, "ser_"), abs(repli_effect_size_value), NA)) %>% 
        mutate(across(contains("conv"), abs))
      
      orig_effects <- orig_outcomes %>% 
        select(claim_id, paper_id,orig_sample_size_value, orig_conv_r, orig_conv_r_lb, orig_conv_r_ub,
               orig_effect_size_type_repli, orig_effect_size_value_repli,pub_year,field) %>%
        semi_join(repli_effects, by = "claim_id") %>% 
        mutate(ser_effect = ifelse(str_detect(orig_effect_size_type_repli, "ser_"), abs(orig_effect_size_value_repli), NA)) %>% 
        mutate(across(contains("conv"), abs))
      
      effects_combined <- merge(repli_effects,orig_effects,by="claim_id",all.x = TRUE,all.y = FALSE)
      
      effects_combined <- effects_combined %>%
        filter(!is.na(repli_conv_r)) %>% 
        filter(!is.na(orig_conv_r)) %>% 
        group_by(paper_id) %>%
        mutate(weight = 1/n())
      
      orig_papers_pearsons_r_papers_median <- 
        bootstrap.clust(data=effects_combined,
                        FUN=function(x) {
                          weighted.median(x$orig_conv_r,x$weight)
                        },
                        clustervar = "paper_id",
                        keepvars=c("repli_conv_r","orig_conv_r","weight"),
                        alpha=.05,tails="two-tailed",iters=iters,
                        digits=2,na.rm = FALSE
        )$formatted.text
      
      repli_papers_pearsons_r_papers_median <- 
        bootstrap.clust(data=effects_combined,
                        FUN=function(x) {
                          weighted.median(x$repli_conv_r,x$weight)
                        },
                        clustervar = "paper_id",
                        keepvars=c("repli_conv_r","orig_conv_r","weight"),
                        alpha=.05,tails="two-tailed",iters=iters,
                        digits=2,na.rm = FALSE
        )$formatted.text
      
      ratio_diff_papers_repli_v_orig <- 
        bootstrap.clust(data=effects_combined,
                        FUN=function(x) {
                          (weighted.median(x$orig_conv_r,x$weight)-weighted.median(x$repli_conv_r,x$weight))/
                            weighted.median(x$orig_conv_r,x$weight)
                        },
                        clustervar = "paper_id",
                        keepvars=c("repli_conv_r","orig_conv_r","weight","paper_id"),
                        alpha=.05,tails="two-tailed",iters=iters,
                        digits=1,na.rm = FALSE,format.percent = TRUE
        )$formatted.text
      
      ratio_sv_median_papers_repli_v_orig <- 
        bootstrap.clust(data=effects_combined,
                        FUN=function(x) {
                          (weighted.median(x$orig_conv_r,x$weight)^2-weighted.median(x$repli_conv_r,x$weight)^2)/
                            weighted.median(x$orig_conv_r,x$weight)^2
                        },
                        clustervar = "paper_id",
                        keepvars=c("repli_conv_r","orig_conv_r","weight","paper_id"),
                        alpha=.05,tails="two-tailed",iters=iters,
                        digits=1,na.rm = FALSE,format.percent = TRUE
        )$formatted.text
      
      
      pearsons_r_papers_orig_v_repli_same_dir <- 
        bootstrap.clust(data=effects_combined[(effects_combined$repli_conv_r>=0)==(effects_combined$repli_conv_r>=0),],
                      FUN=function(x) {
                          weightedCorr(x=x$repli_conv_r,
                                       y=x$orig_conv_r,
                                       method="Pearson",weights=x$weight)
                        },
                      clustervar = "paper_id",
                      keepvars=c("repli_conv_r","orig_conv_r","weight"),
                      alpha=.05,tails="two-tailed",iters=iters,
                      digits=3,na.rm = FALSE
                      )$formatted.text
      
      pearsons_r_claims_orig_v_repli_same_dir <- 
        bootstrap.clust(data=effects_combined[(effects_combined$repli_conv_r>=0)==(effects_combined$repli_conv_r>=0),],
                        FUN=function(x) {
                          weightedCorr(x=x$repli_conv_r,
                                       y=x$orig_conv_r,
                                       method="Pearson")
                        },
                        clustervar = "claim_id",
                        keepvars=c("repli_conv_r","orig_conv_r","weight"),
                        alpha=.05,tails="two-tailed",iters=iters,
                        digits=3,na.rm = FALSE
        )$formatted.text
      
      pearsons_r_claims_orig_v_repli_same_dir_v_year <- 
        bootstrap.clust(data=effects_combined,
                        FUN=function(x) {
                          weightedCorr(x=x$success,
                                       y=x$pub_year,
                                       weight=x$weight,
                                       method="Pearson")
                        },
                        clustervar = "paper_id",
                        keepvars=c("success","pub_year","weight"),
                        alpha=.05,tails="two-tailed",iters=iters,
                        digits=3,na.rm = FALSE
        )$formatted.text
      
      median_r_by_year <- 
        do.call(rbind,lapply(2009:2018,function(year){
          data <- effects_combined[effects_combined$pub_year==year,]
          median.orig <- median(data$orig_conv_r)
          median.repli <- median(data$repli_conv_r)
          median.diff <- abs(median.orig-median.repli)
          data.frame(year,median.orig,median.repli,median.diff)
        }))
      
      year_pearsons_r_median_diff_min <- 
        median_r_by_year[median_r_by_year$median.diff==min(median_r_by_year$median.diff),]$year
      year_pearsons_r_median_diff_max <- 
        median_r_by_year[median_r_by_year$median.diff==max(median_r_by_year$median.diff),]$year
      pearsons_r_orig_median_diff_yearmin <- 
        format.round(
          median_r_by_year[median_r_by_year$median.diff==min(median_r_by_year$median.diff),]$median.orig,2)
      pearsons_r_repli_median_diff_yearmin <- 
        format.round(
          median_r_by_year[median_r_by_year$median.diff==min(median_r_by_year$median.diff),]$median.repli,2)
      pearsons_r_orig_median_diff_yearmax <- 
        format.round(
          median_r_by_year[median_r_by_year$median.diff==max(median_r_by_year$median.diff),]$median.orig,2)
      pearsons_r_repli_median_diff_yearmax <- 
        format.round(
          median_r_by_year[median_r_by_year$median.diff==max(median_r_by_year$median.diff),]$median.repli,2)
      
      
      chisq.years <- wtd.chi.sq(effects_combined$success,effects_combined$pub_year,weight=effects_combined$weight)
      chisq_success_years_pvalue <- format.round(chisq.years[3],2)
    }
    
    # Results: Original and replication effects by new data or secondary data replication
    {
      repli_effects <- repli_outcomes %>%
        filter(!is_covid) %>%
        filter(repli_version_of_record) %>%
        select(report_id, claim_id,repli_sample_size_value, success = repli_score_criteria_met, repli_type,
               repli_conv_r, repli_conv_r_lb, repli_conv_r_ub, repli_effect_size_type, repli_effect_size_value) %>%
        mutate(ser_effect = ifelse(str_detect(repli_effect_size_type, "ser_"), abs(repli_effect_size_value), NA)) %>%
        mutate(across(contains("conv"), abs))
      
      orig_effects <- orig_outcomes %>%
        semi_join(repli_effects, by = "claim_id") %>%
        select(claim_id, paper_id,orig_sample_size_value, orig_conv_r, orig_conv_r_lb, orig_conv_r_ub,
               orig_effect_size_type_repli, orig_effect_size_value_repli) %>%
        mutate(ser_effect = ifelse(str_detect(orig_effect_size_type_repli, "ser_"), abs(orig_effect_size_value_repli), NA)) %>%
        mutate(across(contains("conv"), abs)) %>%
        mutate(paper_id = select(., claim_id) %>% apply(1, function(x) str_split(x, "_") %>% unlist() %>% first())) %>%
        left_join(paper_metadata %>% select(paper_id, field = COS_pub_category), by = "paper_id")
      
      effects_combined <- merge(orig_effects,repli_effects,by="claim_id",all.x = TRUE,all.y = FALSE)
      
      effects_combined <- effects_combined %>%
        group_by(paper_id) %>%
        mutate(weight = 1/n()) %>%
        ungroup()
      
      ratio_papers_repli_stat_sig_same_pattern_new_data_vs_secondary <- 
        bootstrap.clust(data=effects_combined,
                      FUN=function(x) {
                        (sum(x[x$repli_type=="new data",]$success*x[x$repli_type=="new data",]$weight)/length(unique(x[x$repli_type=="new data",]$paper_id)))/
                          (sum(x[x$repli_type=="secondary data",]$success*x[x$repli_type=="secondary data",]$weight)/length(unique(x[x$repli_type=="secondary data",]$paper_id)))
                      },
                      clustervar = "paper_id",
                      keepvars=c("repli_type","success","weight","paper_id"),
                      alpha=.05,tails="two-tailed",iters=iters,
                      digits=3,na.rm = FALSE
      )$formatted.text
      
      ratio_claims_repli_stat_sig_same_pattern_new_data_vs_secondary <- 
        bootstrap.clust(data=effects_combined,
                        FUN=function(x) {
                          (sum(x[x$repli_type=="new data",]$success)/length(unique(x[x$repli_type=="new data",]$claim_id)))/
                            (sum(x[x$repli_type=="secondary data",]$success)/length(unique(x[x$repli_type=="secondary data",]$claim_id)))
                        },
                        clustervar = "claim_id",
                        keepvars=c("repli_type","success","weight","paper_id"),
                        alpha=.05,tails="two-tailed",iters=iters,
                        digits=3,na.rm = FALSE
        )$formatted.text
      
      effects_combined <- effects_combined %>%
        filter(!is.na(repli_conv_r) & !is.na(orig_conv_r)) %>%
        group_by(paper_id) %>%
        mutate(weight = 1/n()) %>%
        ungroup()
      
      secondary_repli_by_field <- na.omit(do.call(rbind,lapply(unique(repli_outcomes_merged$field),function(field) {
        data <- repli_outcomes_merged[!is.na(repli_outcomes_merged$field) & repli_outcomes_merged$field==field,]
        data <- data %>% group_by(paper_id) %>% mutate(weight = 1/n()) %>% ungroup()
        n_secondary_data <- sum(as.integer(data$repli_type=="secondary data")*data$weight)
        n_total <- length(unique(data$paper_id))
        p <- n_secondary_data/n_total
        p.formatted <- paste0(format.round(100*p,1),"%")
        p.formatted.combined <- paste0(p.formatted,", n=",n_secondary_data,"/",n_total,"")
        data.frame(field,n_secondary_data,n_total,p,p.formatted,p.formatted.combined)
      })))
      
      secondary_repli_by_field <- secondary_repli_by_field[order(-secondary_repli_by_field$p),]

      for (i in 1:nrow(secondary_repli_by_field)){
        assign(paste0("field_",i,"_highest_p_secondary_data"),secondary_repli_by_field$field[i])
        assign(paste0("p_",i,"_highest_p_secondary_data"),secondary_repli_by_field$p.formatted.combined[i])
      }
      
      repli_outcomes_nd <- repli_outcomes[repli_outcomes$repli_type=="new data",] %>%
        group_by(paper_id) %>%
        mutate(weight = 1/n()) %>%
        ungroup()
      
      repli_outcomes_sd <- repli_outcomes[repli_outcomes$repli_type=="secondary data",] %>%
        group_by(paper_id) %>%
        mutate(weight = 1/n()) %>%
        ungroup()
      
      median_power_100_effect_new_data_papers <- 
        paste0(format.round(100*weighted.median(repli_outcomes_nd$rr_power_100_original_effect,repli_outcomes_nd$weight,na.rm=TRUE),1),"%")
      median_power_100_effect_new_data_claims <- 
        paste0(format.round(100*median(repli_outcomes_nd$rr_power_100_original_effect,na.rm=TRUE),1),"%")
      
      median_power_100_effect_secondary_data_papers <- 
        paste0(format.round(100*weighted.median(repli_outcomes_sd$rr_power_100_original_effect,repli_outcomes_sd$weight,na.rm=TRUE),1),"%")
      median_power_100_effect_secondary_data_claims <- 
        paste0(format.round(100*median(repli_outcomes_sd$rr_power_100_original_effect,na.rm=TRUE),1),"%")
      
      mean_power_100_effect_new_data_papers <- 
        paste0(format.round(100*weighted.mean(repli_outcomes_nd$rr_power_100_original_effect,repli_outcomes_nd$weight,na.rm=TRUE),1),"%")
      mean_power_100_effect_new_data_claims <- 
        paste0(format.round(100*mean(repli_outcomes_nd$rr_power_100_original_effect,na.rm=TRUE),1),"%")
      
      mean_power_100_effect_secondary_data_papers <- 
        paste0(format.round(100*weighted.mean(repli_outcomes_sd$rr_power_100_original_effect,repli_outcomes_sd$weight,na.rm=TRUE),1),"%")
      mean_power_100_effect_secondary_data_claims <- 
        paste0(format.round(100*mean(repli_outcomes_sd$rr_power_100_original_effect,na.rm=TRUE),1),"%")
      
      effects_combined_nd <- effects_combined[effects_combined$repli_type=="new data",] %>%
        group_by(paper_id) %>%
        mutate(weight = 1/n()) %>%
        ungroup()
      n_new_data_papers <- length(unique(effects_combined_nd$paper_id))
      n_new_data_papers_repli_r_smaller <-
        sum(as.integer(effects_combined_nd$repli_conv_r<effects_combined_nd$orig_conv_r)*
              effects_combined_nd$weight)
      n_new_data_papers_repli_r_smaller_formatted <- format.round(n_new_data_papers_repli_r_smaller,1)
      p_new_data_papers_repli_r_smaller <- paste0(format.round(
        100*n_new_data_papers_repli_r_smaller/n_new_data_papers,1
      ),"%")
      
      effects_combined_sd <- effects_combined[effects_combined$repli_type=="secondary data",] %>%
        group_by(paper_id) %>%
        mutate(weight = 1/n()) %>%
        ungroup()
      n_secondary_data_papers <- length(unique(effects_combined_sd$paper_id))
      n_secondary_data_papers_repli_r_smaller <-
        sum(as.integer(effects_combined_sd$repli_conv_r<effects_combined_sd$orig_conv_r)*
              effects_combined_sd$weight)
      n_secondary_data_papers_repli_r_smaller_formatted <- format.round(n_secondary_data_papers_repli_r_smaller,1)
      p_secondary_data_papers_repli_r_smaller <- paste0(format.round(
        100*n_secondary_data_papers_repli_r_smaller/n_secondary_data_papers,1
      ),"%")
      
      all_effects_nd <- repli_outcomes %>% 
        filter(!is_covid) %>% 
        filter(repli_version_of_record) %>% 
        filter(repli_type=="new data") %>% 
        select(paper_id,report_id, claim_id, repli_pattern_criteria_met, success = repli_score_criteria_met,
               repli_conv_r, repli_conv_r_lb, repli_conv_r_ub) %>% 
        left_join(
          orig_outcomes %>% 
            select(claim_id, orig_conv_r, orig_conv_r_lb, orig_conv_r_ub) %>% 
            distinct(),
          by = "claim_id"
        ) %>% 
        drop_na(repli_conv_r) %>% 
        drop_na(orig_conv_r) %>% 
        mutate(across(contains("orig"), abs)) %>% 
        mutate(
          across(
            contains("repli"),
            function(x) ifelse(repli_pattern_criteria_met, abs(x), -1*abs(x))
          )
        ) %>%
        group_by(paper_id) %>%
        mutate(weight = 1/n()) %>%
        ungroup()
      
      n_claims_repli_nd <- nrow(all_effects_nd)
      
      all_effects_sd <- repli_outcomes %>% 
        filter(!is_covid) %>% 
        filter(repli_version_of_record) %>% 
        filter(repli_type=="secondary data") %>% 
        select(paper_id,report_id, claim_id, repli_pattern_criteria_met, success = repli_score_criteria_met,
               repli_conv_r, repli_conv_r_lb, repli_conv_r_ub) %>% 
        left_join(
          orig_outcomes %>% 
            select(claim_id, orig_conv_r, orig_conv_r_lb, orig_conv_r_ub) %>% 
            distinct(),
          by = "claim_id"
        ) %>% 
        drop_na(repli_conv_r) %>% 
        drop_na(orig_conv_r) %>% 
        mutate(across(contains("orig"), abs)) %>% 
        mutate(
          across(
            contains("repli"),
            function(x) ifelse(repli_pattern_criteria_met, abs(x), -1*abs(x))
          )
        ) %>%
        group_by(paper_id) %>%
        mutate(weight = 1/n()) %>%
        ungroup()
      
      n_claims_repli_sd <- nrow(all_effects_sd)
      
     
      
    }
    
    # Sampling frame and selection of claims for replication
    {
      
      n_journals_SCORE_all <- paper_metadata_orig %>%
        filter(!is_covid) %>%
        count(publication_standard) %>%
        nrow()
      
      n_papers_selected_repli <- paper_status_tracking %>% filter(RR) %>% nrow()
      
      n_papers_initial_sample <- paper_status_tracking %>% filter(p1_delivery | p2_delivery) %>% nrow()
      
      n_repli_papers_eligible <- paper_status_tracking %>% filter(RR | p2_delivery) %>% nrow()
      
      n_papers_p2_constrained <-  paper_status_tracking %>% filter(p2_delivery) %>% nrow()
      
      n_repli_papers_eligible_single_claim <- paper_status_tracking %>%
        filter(RR | p2_delivery) %>%
        anti_join(paper_status_tracking %>% filter(bushel), by = "paper_id") %>%
        nrow()
      
      n_repli_papers_eligible_multi_claim <- paper_status_tracking %>% filter(bushel) %>% nrow()
      
      n_repli_hybrid <- repli_outcomes_orig %>%
        filter(!is_covid) %>%
        filter(repli_type == "original and secondary data") %>%
        nrow()
      
      n_total_replis_conducted <- repli_outcomes_orig %>%
        filter(!is_covid & repli_type != "original and secondary data") %>%
        filter(is.na(manylabs_type) | manylabs_type != "aggregation") %>%
        mutate(rep_attempt = select(., c(rr_id, claim_id)) %>% apply(1, function(x) str_c(x, collapse = " -- "))) %>%
        count(rep_attempt) %>%
        nrow()
      
      n_repli_attempt_started <- rr_internal_tracking %>%
        filter(field != "covid") %>%
        filter(str_detect(type, "Replication")) %>%
        pull(rr_id) %>% unique() %>% length()
      
      n_repli_claims_same_protocol <- repli_outcomes_orig %>%
        filter(!is_covid) %>%
        filter(is_manylabs == "traditional") %>%
        count(claim_id) %>%
        nrow()
      
      n_repli_claims_distinct_protocol <- repli_outcomes_orig %>%
        filter(!is_covid) %>%
        filter(is_manylabs == "posthoc") %>%
        count(claim_id) %>%
        nrow()
      
      p_repli_attempts_selected_p1 <- repli_outcomes %>%
        mutate(p1 = paper_id %in% (paper_status_tracking %>% filter(p1_delivery) %>% pull(paper_id))) %>%
        count(p1) %>%
        mutate(prop = (100*(n/sum(n))) %>% round(2)) %>%
        filter(p1) %>%
        pull(prop) %>%
        format.round(1)
        
    }

  }
  
  # Supplement stats
  {
    # Tables
    {
      # Table S3
      {
        table_s3 <- rr_internal_tracking %>%
          filter(field != "covid") %>%
          filter(str_detect(type, "Replication")) %>%
          mutate(Completed = ifelse( outcome | results_available == "yes", "Yes", "No")) %>%
          mutate(`Replication data` = ifelse(rr_id %in% repli_outcomes$rr_id, "Yes", "No")) %>%
          select(`Paper ID` = paper_id, `Project ID` = rr_id, `OSF` = project_guid, Completed, `Replication data`) %>%
          arrange(`Paper ID`)

        for (row in 1:nrow(table_s3)){
          for (col in 1:ncol(table_s3)){
            assign(paste0("table_s3_",row,"_",col),
                   as.character(table_s3[row,col][1,1]))
          }
        }

      }
      
      # Table S6
      {
        # repli_outcomes <- repli_outcomes %>%
        #   filter(!is_covid & repli_version_of_record)
        # orig_outcomes <- orig_outcomes %>%
        #   semi_join(repli_outcomes, by = "claim_id") %>%
        orig_measures <- orig_outcomes %>%
          semi_join(repli_outcomes, by = "claim_id") %>%
          mutate(
            orig_es_native_type = case_when(
              orig_effect_size_type_repro == "beta" ~ NA_character_,
              orig_effect_size_type_repro %in% c("cohen_d", "cohen_dz") ~ "Cohen's d",
              orig_effect_size_type_repro == "cohen_f_squared" ~ "Cohen's f squared",
              orig_effect_size_type_repro == "cohen_q" ~ "Cohen's q",
              orig_effect_size_type_repro == "cramer_v" ~ "Cramer's V",
              orig_effect_size_type_repro == "difference_in_coefficients" ~ NA_character_,
              orig_effect_size_type_repro %in% c("eta_squared", "partial_eta_squared") ~ "(Partial) eta squared",
              orig_effect_size_type_repro == "hazard_ratio" ~ "Hazard ratio",
              orig_effect_size_type_repro == "incidence_rate_ratio" ~ "Incident rate ratio",
              orig_effect_size_type_repro == "log_odds_ratio" ~ NA_character_,
              orig_effect_size_type_repro == "odds_ratio" ~ "Odds ratio",
              orig_effect_size_type_repro == "path_coefficient" ~ NA_character_,
              orig_effect_size_type_repro == "pearson_r" ~ "Correlation",
              orig_effect_size_type_repro == "phi" ~ "Cramer's V",
              orig_effect_size_type_repro == "regression_coefficient" ~ NA_character_,
              orig_effect_size_type_repro == "relative_risk_ratio" ~ "Relative rate ratio",
              orig_effect_size_type_repro %in% c("ser_method", "ser_method_t") ~ NA_character_
            )
          ) %>%
          mutate(orig_effect_size_value_repro = ifelse(is.na(orig_es_native_type), NA, orig_effect_size_value_repro)) %>%
          select(claim_id, `Original sample size value` = orig_sample_size_value, `Original sample size units` = orig_sample_size_units,
                 `Original effect size type` = orig_es_native_type, `Original effect size value` = orig_effect_size_value_repro,
                 `Original effect size (partial correlation)` = orig_conv_r)
        repli_measures <- repli_outcomes %>%
          mutate(
            repli_es_native_type = case_when(
              repli_effect_size_type_raw %in% c("cohen_d", "cohen_dz") ~ "Cohen's d",
              repli_effect_size_type_raw == "cohen_f_squared" ~ "Cohen's f squared",
              repli_effect_size_type_raw == "cohen_q" ~ "Cohen's q",
              repli_effect_size_type_raw %in% c("cohen_w", "cramer_v", "phi") ~ "Cramer's V",
              repli_effect_size_type_raw %in% c("eta_squared", "partial_eta_squared") ~ "(Partial) eta squared",
              repli_effect_size_type_raw == "hazard_ratio" ~ "Hazard ratio",
              repli_effect_size_type_raw %in% c("log_odds_ratio", "ser_method", "ordered_log_odds_ratio") ~ NA_character_,
              repli_effect_size_type_raw %in% c("partial_correlation", "pearson_r", "rank_bis") ~ "Correlation",
              repli_effect_size_type_raw == "RRR" ~ "Relative risk reduction"
            )
          ) %>%
          mutate(repli_es_native_value = ifelse(is.na(repli_es_native_type), NA, repli_effect_size_value_raw)) %>%
          select(claim_id, `Replication sample size value` = repli_sample_size_value,
                 `Replication sample size units` = repli_sample_size_units, `Traditional power` = repli_power_for_75_effect,
                 `Design power` = rr_power_75_original_effect_design,
                 `Combined power` = combined_power_75,
                 `Replication effect size units` = repli_es_native_type, `Replication effect size value` = repli_es_native_value,
                 `Replication effect size (partial correlation)` = repli_conv_r)
        table_s6 <- orig_measures %>%
          left_join(repli_measures, by = "claim_id") %>%
          mutate(
            across(
              c(contains("effect size value"), contains("correlation"), contains("power")),
              function(x) round(x, 2)
            )
          ) %>%
          rename(Claim = `claim_id`)
      }
      
      # Table S7: Selected ratios of effect sizes for original and replication studies in common units compared to Pearson’s r partial correlation conversions
      {
        # get all effect sizes in the correct sign format
        pared_down <- repli_outcomes_orig %>% 
          filter(!is_covid & repli_version_of_record) %>% 
          left_join(
            orig_outcomes %>% select(claim_id, orig_type = orig_effect_size_type_repli, 
                                     orig_value = orig_effect_size_value_repli, orig_conv_r),
            by = "claim_id"
          ) %>% 
          mutate(orig_conv_r = abs(orig_conv_r), repli_conv_r = abs(repli_conv_r)) %>% 
          mutate(repli_conv_r = ifelse(repli_pattern_criteria_met, repli_conv_r, (-1)*repli_conv_r)) %>% 
          rename(repli_value = repli_effect_size_value)
        
        ## cohen's f-squared
        cohen_f_squared <- pared_down %>% 
          filter(str_detect(repli_effect_size_type, "cohen_f") & str_detect(orig_type, "cohen_f")) %>% 
          select(report_id, repli_value, repli_conv_r, orig_value, orig_conv_r) %>% 
          dplyr::summarize(
            n = n(),
            ratio_native = mean(repli_value/orig_value),
            ratio_conv = mean(repli_conv_r/orig_conv_r)
          ) %>% 
          mutate(method = "Cohen's f-squared", .before = n)
        
        ## cohen's d
        cohen_d <- pared_down %>% 
          filter(str_detect(repli_effect_size_type, "d_sample") & str_detect(orig_type, "d_sample")) %>% 
          select(report_id, repli_value, repli_conv_r, orig_value, orig_conv_r) %>% 
          dplyr::summarize(
            n = n(),
            ratio_native = mean(repli_value/orig_value),
            ratio_conv = mean(repli_conv_r/orig_conv_r)
          ) %>% 
          mutate(method = "Cohen's d", .before = n)
        
        ## correlation
        cor_all <- pared_down %>% 
          filter(str_detect(repli_effect_size_type, "bis|part_cor|pearson") & str_detect(orig_type, "bis|part_cor|pearson")) %>% 
          select(report_id, repli_value, repli_conv_r, orig_value, orig_conv_r) %>% 
          dplyr::summarize(
            n = n(),
            ratio_native = mean(repli_value/orig_value),
            ratio_conv = mean(repli_conv_r/orig_conv_r)
          ) %>% 
          mutate(method = "Correlation", .before = n)
        
        
        table_s7 <- cohen_f_squared %>% 
          bind_rows(cohen_d) %>% 
          bind_rows(cor_all) %>% 
          mutate(across(c(ratio_native, ratio_conv), function(x) round(x, 2)))
        
        for (row in 1:nrow(table_s7)){
          for (col in 1:ncol(table_s7)){
            assign(paste0("table_s7_",row,"_",col),
                   table_s7[[row,col]])
          }
        }
        
        rm(cor_all,cohen_d,pared_down,cohen_f_squared)
      }
      
      # Table S8: Papers and claims selected, and replication attempts, by year of publication.
      {
        fields.order <- sort(c("Psychology And Health","Business","Sociology And Criminology",
                               "Economics And Finance","Political Science","Education"))
        
        format.row <- function(data){
          data <- data %>% 
            select(paper_id) %>% 
            left_join(paper_metadata %>% select(paper_id, pub_year), by = "paper_id") %>% 
            count(pub_year)
          total <- sum(data$n)
          pub_year <- "Total"
          n <- paste0(total," (100%)")
          data$n <- paste0(data$n," (",
                           format.round(100*data$n/sum(data$n),1),
                           "%)")
          data <- rbind(data,data.frame(pub_year,n))
          colnames <- data$pub_year
          data <- data.frame(t(rbind(data,data.frame(pub_year,n))[,-1]))
          colnames(data) <- colnames
          data[c(as.character(2009:2018),"Total")]
        }
        n.row <- function(data){
          data <- data %>% 
            select(paper_id) %>% 
            left_join(paper_metadata %>% select(paper_id, pub_year), by = "paper_id") %>% 
            count(pub_year)
          total <- sum(data$n)
          pub_year <- "Total"
          n <- total
          data <- rbind(data,data.frame(pub_year,n))
          colnames <- data$pub_year
          data <- data.frame(t(rbind(data,data.frame(pub_year,n))[,-1]))
          colnames(data) <- colnames
          data[c(as.character(2009:2018),"Total")]
        }
        p.row <- function(data){
          data <- data %>% 
            select(paper_id) %>% 
            left_join(paper_metadata %>% select(paper_id, pub_year), by = "paper_id") %>% 
            count(pub_year)
          total <- sum(data$n)
          pub_year <- "Total"
          n <- 1
          data$n <- data$n/sum(data$n)
          data <- rbind(data,data.frame(pub_year,n))
          colnames <- data$pub_year
          data <- data.frame(t(rbind(data,data.frame(pub_year,n))[,-1]))
          colnames(data) <- colnames
          data[c(as.character(2009:2018),"Total")]
        }
        
        r1.data <- paper_status_tracking %>% filter(p1_delivery | p2_delivery)
        r2.data <- paper_status_tracking %>% filter(RR | p2_delivery)
        r3.data <- paper_status_tracking %>% filter(bushel)
        r4.data <- paper_status_tracking %>% filter((p2_delivery | RR) & !bushel)
        r5.data <- rr_internal_tracking %>% 
          filter(str_detect(type, "Replication")) %>%
          select(paper_id) %>% 
          distinct() %>% 
          semi_join(paper_status_tracking %>% filter(p1_delivery | p2_delivery), by = "paper_id")
        r6.data <-  repli_outcomes_orig %>% 
          filter(repli_type != "original and secondary data") %>% 
          filter(!is_covid) %>% 
          filter(repli_version_of_record) %>% 
          select(paper_id) %>% 
          distinct()
        r7.data <- repli_outcomes_orig %>%
          filter(repli_type != "original and secondary data") %>%
          semi_join(paper_status_tracking %>% filter(p1_delivery|p2_delivery), by = "paper_id") %>%
          filter(is.na(manylabs_type) | manylabs_type != "aggregation") %>%
          select(paper_id, claim_id, rr_id) %>%
          distinct()
        r8.data <-  repli_outcomes_orig %>%
          filter(repli_type != "original and secondary data") %>% 
          filter(!is_covid) %>% 
          filter(repli_version_of_record) %>% 
          select(paper_id, claim_id) %>% 
          distinct()
        
        for (i in 1:8){
          assign(paste0("r",i),format.row(get(paste0("r",i,".data"))))
          assign(paste0("r",i,".n"),n.row(get(paste0("r",i,".data"))))
          assign(paste0("r",i,".p"),p.row(get(paste0("r",i,".data"))))
        }
        
        table_s8 <- rbind(r1,r2,r3,r4,r5,r6,r7,r8)
        table_s8_n <- rbind(r1.n,r2.n,r3.n,r4.n,r5.n,r6.n,r7.n,r8.n)
        table_s8_p <- rbind(r1.p,r2.p,r3.p,r4.p,r5.p,r6.p,r7.p,r8.p)
        for (row in 1:nrow(table_s8)){
          for (col in 1:ncol(table_s8)){
            assign(paste0("table_s8_",row,"_",col),
                   table_s8[row,col])
          }
        }
        rm(r1,r2,r3,r4,r5,r6,r7,r8)
      }
      
      # Table S9: Paper-level count of replications completed from Phase 1 and Phase 2 by discipline
      {
        table_s9 <- repli_outcomes_orig %>% 
          filter(!is_covid & repli_version_of_record) %>% 
          left_join(paper_metadata %>% select(paper_id, field = COS_pub_category), by = "paper_id") %>% 
          select(paper_id, field, type_internal) %>% 
          mutate(type_internal = ifelse(type_internal == "bushel", "p1", type_internal)) %>% 
          distinct() %>% 
          group_by(field, type_internal) %>% 
          dplyr::summarize(t = n()) %>% 
          pivot_wider(names_from = type_internal, values_from = t) %>% 
          ungroup() %>% 
          mutate(field = str_to_sentence(field)) %>% 
          bind_rows(summarize_at(., vars(-field), sum)) %>% 
          replace_na(., list(field = "Total"))
        
        for (row in 1:nrow(table_s9)){
          for (col in 1:ncol(table_s9)){
            assign(paste0("table_s9_",row,"_",col),
                   table_s9[[row,col]])
          }
        }
        
      }
      
      # Table S10: Original and replication findings by Pearson's r effect size by papers and claims
      {
        repli_effects <- repli_outcomes %>% 
          filter(type_internal!="p2") %>% 
          filter(!is_covid) %>% 
          filter(repli_version_of_record) %>% 
          select(report_id, claim_id, repli_sample_size_value, success = repli_score_criteria_met, repli_type,
                 repli_conv_r, repli_conv_r_lb, repli_conv_r_ub, repli_effect_size_type, repli_effect_size_value) %>%
          mutate(ser_effect = ifelse(str_detect(repli_effect_size_type, "ser_"), abs(repli_effect_size_value), NA)) %>% 
          mutate(across(contains("conv"), abs))
        
        orig_effects <- orig_outcomes %>% 
          select(claim_id, paper_id,orig_sample_size_value, orig_conv_r, orig_conv_r_lb, orig_conv_r_ub,
                 orig_effect_size_type_repli, orig_effect_size_value_repli) %>%
          semi_join(repli_effects, by = "claim_id") %>% 
          mutate(ser_effect = ifelse(str_detect(orig_effect_size_type_repli, "ser_"), abs(orig_effect_size_value_repli), NA)) %>% 
          mutate(across(contains("conv"), abs))
        
        effects_combined <- merge(repli_effects,orig_effects,by="claim_id",all.x = TRUE,all.y = FALSE)
        
        effects_combined <- effects_combined %>%
          filter(!is.na(repli_conv_r)) %>%
          filter(!is.na(orig_conv_r)) %>%
          group_by(paper_id) %>%
          mutate(weight = 1/n())
        
        
        # Number of papers and claims
        r1 <- c(length(unique(effects_combined$paper_id)),length(unique(effects_combined$paper_id)),
                length(unique(effects_combined$claim_id)),length(unique(effects_combined$claim_id)))
        
        # Number of effect sizes
        r2 <- c(sum(as.numeric(!is.na(effects_combined$orig_conv_r))*effects_combined$weight),
                sum(as.numeric(!is.na(effects_combined$repli_conv_r))*effects_combined$weight),
                sum(!is.na(effects_combined$orig_conv_r)),
                sum(!is.na(effects_combined$repli_conv_r))
        )
        
        # Median/IQR of sample sizes
        r3.median <- c(weighted.quantile(effects_combined$orig_sample_size_value,effects_combined$weight,.5),
                       weighted.quantile(effects_combined$repli_sample_size_value,effects_combined$weight,.5),
                       quantile(effects_combined$orig_sample_size_value,.5,na.rm=TRUE),
                       quantile(effects_combined$repli_sample_size_value,.5,na.rm=TRUE)
        )
        
        r3.IQR <- c(format.round(weighted.quantile(effects_combined$orig_sample_size_value,effects_combined$weight,.50) - weighted.quantile(effects_combined$orig_sample_size_value,effects_combined$weight,.25),1),
                    format.round(weighted.quantile(effects_combined$repli_sample_size_value,effects_combined$weight,.50) - weighted.quantile(effects_combined$repli_sample_size_value,effects_combined$weight,.25),1),
                    format.round(IQR(effects_combined$orig_sample_size_value,na.rm=TRUE),1),
                    format.round(IQR(as.integer(effects_combined$repli_sample_size_value),na.rm=TRUE),1)
        )
        
        r3 <- paste0(r3.median,"\n[",r3.IQR,"]")
        
        # Pearson's R effect size
        r4.median <- c(weighted.quantile(effects_combined$orig_conv_r,effects_combined$weight,.5),
                       weighted.quantile(effects_combined$repli_conv_r,effects_combined$weight,.5),
                       quantile(effects_combined$orig_conv_r,.5),
                       quantile(effects_combined$repli_conv_r,.5)
        )
        
        r4.median.formatted <- format.round(r4.median,2)
        
        r4.SD <- c(sqrt(wtd.var(effects_combined$orig_conv_r,effects_combined$weight)),
                   sqrt(wtd.var(effects_combined$repli_conv_r,effects_combined$weight)),
                   sd(effects_combined$orig_conv_r,1),
                   sd(effects_combined$repli_conv_r,1)
        )
        r4.SD.formatted <- format.round(r4.SD,2)
        
        r4 <- paste0(r4.median.formatted,"\n(",r4.SD.formatted,")")
        
        table_s10 <- rbind(r1,r2,r3,r4)
        for (row in 1:nrow(table_s10)){
          for (col in 1:ncol(table_s10)){
            assign(paste0("table_s10_",row,"_",col),
                   table_s10[row,col])
            if(row==4){
              assign(paste0("table_s10_median_",row,"_",col),
                     r4.median[col])
              assign(paste0("table_s10_SD_",row,"_",col),
                     r4.SD[col])
              assign(paste0("table_s10_median_formatted_",row,"_",col),
                     r4.median.formatted[col])
              assign(paste0("table_s10_SD_formatted_",row,"_",col),
                     r4.SD.formatted[col])
            }
          }
        }
        rm(r1,r2,r3,r4)
      }
      
      # Table S11: Original and replication findings by Pearson's r effect size by papers and claims.
      {
        repli_effects <- repli_outcomes %>% 
          filter(type_internal=="p2") %>% 
          filter(!is_covid) %>% 
          filter(repli_version_of_record) %>% 
          select(report_id, claim_id, repli_sample_size_value, success = repli_score_criteria_met, repli_type,
                 repli_conv_r, repli_conv_r_lb, repli_conv_r_ub, repli_effect_size_type, repli_effect_size_value) %>%
          mutate(ser_effect = ifelse(str_detect(repli_effect_size_type, "ser_"), abs(repli_effect_size_value), NA)) %>% 
          mutate(across(contains("conv"), abs))
        
        orig_effects <- orig_outcomes %>% 
          select(claim_id, paper_id,orig_sample_size_value, orig_conv_r, orig_conv_r_lb, orig_conv_r_ub,
                 orig_effect_size_type_repli, orig_effect_size_value_repli) %>%
          semi_join(repli_effects, by = "claim_id") %>% 
          mutate(ser_effect = ifelse(str_detect(orig_effect_size_type_repli, "ser_"), abs(orig_effect_size_value_repli), NA)) %>% 
          mutate(across(contains("conv"), abs))
        
        effects_combined <- merge(repli_effects,orig_effects,by="claim_id",all.x = TRUE,all.y = FALSE)
        
        effects_combined <- effects_combined %>%
          filter(!is.na(repli_conv_r)) %>%
          filter(!is.na(orig_conv_r)) %>%
          group_by(paper_id) %>%
          mutate(weight = 1/n())
        
        
        # Number of papers and claims
        r1 <- c(length(unique(effects_combined$paper_id)),length(unique(effects_combined$paper_id)),
                length(unique(effects_combined$claim_id)),length(unique(effects_combined$claim_id)))
        
        # Number of effect sizes
        r2 <- c(sum(as.numeric(!is.na(effects_combined$orig_conv_r))*effects_combined$weight),
                sum(as.numeric(!is.na(effects_combined$repli_conv_r))*effects_combined$weight),
                sum(!is.na(effects_combined$orig_conv_r)),
                sum(!is.na(effects_combined$repli_conv_r))
        )
        
        # Median/IQR of sample sizes
        r3.median <- c(weighted.quantile(effects_combined$orig_sample_size_value,effects_combined$weight,.5),
                       weighted.quantile(effects_combined$repli_sample_size_value,effects_combined$weight,.5),
                       quantile(effects_combined$orig_sample_size_value,.5,na.rm=TRUE),
                       quantile(effects_combined$repli_sample_size_value,.5,na.rm=TRUE)
        )
        
        r3.IQR <- c(format.round(weighted.quantile(effects_combined$orig_sample_size_value,effects_combined$weight,.50) - weighted.quantile(effects_combined$orig_sample_size_value,effects_combined$weight,.25),1),
                    format.round(weighted.quantile(effects_combined$repli_sample_size_value,effects_combined$weight,.50) - weighted.quantile(effects_combined$repli_sample_size_value,effects_combined$weight,.25),1),
                    format.round(IQR(effects_combined$orig_sample_size_value,na.rm=TRUE),1),
                    format.round(IQR(as.integer(effects_combined$repli_sample_size_value),na.rm=TRUE),1)
        )
        
        r3 <- paste0(r3.median,"\n[",r3.IQR,"]")
        
        # Pearson's R effect size
        r4.median <- c(weighted.quantile(effects_combined$orig_conv_r,effects_combined$weight,.5),
                       weighted.quantile(effects_combined$repli_conv_r,effects_combined$weight,.5),
                       quantile(effects_combined$orig_conv_r,.5),
                       quantile(effects_combined$repli_conv_r,.5)
        )
        
        r4.median.formatted <- format.round(r4.median,2)
        
        r4.SD <- c(sqrt(wtd.var(effects_combined$orig_conv_r,effects_combined$weight)),
                   sqrt(wtd.var(effects_combined$repli_conv_r,effects_combined$weight)),
                   sd(effects_combined$orig_conv_r,1),
                   sd(effects_combined$repli_conv_r,1)
        )
        r4.SD.formatted <- format.round(r4.SD,2)
        
        r4 <- paste0(r4.median.formatted,"\n(",r4.SD.formatted,")")
        
        table_s11 <- rbind(r1,r2,r3,r4)
        for (row in 1:nrow(table_s11)){
          for (col in 1:ncol(table_s11)){
            assign(paste0("table_s11_",row,"_",col),
                   table_s11[row,col])
            if(row==4){
              assign(paste0("table_s11_median_",row,"_",col),
                     r4.median[col])
              assign(paste0("table_s11_SD_",row,"_",col),
                     r4.SD[col])
              assign(paste0("table_s11_median_formatted_",row,"_",col),
                     r4.median.formatted[col])
              assign(paste0("table_s11_SD_formatted_",row,"_",col),
                     r4.SD.formatted[col])
            }
          }
        }
        rm(r1,r2,r3,r4)
      }
    
      # Table S12: Replication success rates for binary assessments by claims
      {
        all_cases <- repli_binary %>% 
          left_join(repli_outcomes_orig %>% select(report_id, score = repli_score_criteria_met), by = "report_id") %>% 
          rename_with(., function(x) str_remove_all(x, "repli_binary_"), .cols = everything()) %>% 
          pivot_longer(cols = c(-report_id,-paper_id,-claim_id), names_to = "metric", values_to = "outcome") %>% 
          drop_na() %>% 
          group_by(metric) %>% 
          dplyr::summarize(
            success = sum(outcome),
            tot = n(),
            Rate = glue("{100*round(success/tot, 2)}%")
          ) %>% 
          mutate(Counts = glue("{success} of {tot}")) %>% 
          select(metric, Counts, Rate) %>% 
          mutate(
            metric = case_match(
              metric,
              "analyst" ~ "Analyst interpretation",
              "score" ~ "Sig + pattern",
              "sum_p" ~ "Sum of p-values",
              "skep_p" ~ "Skeptical p-value",
              "orig_wthn" ~ "Orig. in rep. CI",
              "rep_wthn" ~ "Rep. in orig. CI",
              "wthn_pi" ~ "Rep. in prediction interval",
              "meta_success" ~ "Meta-analysis",
              "bma_result" ~ "Bayesian meta-analysis",
              "telescopes" ~ "Small telescopes",
              "bf_result" ~ "Bayes factor",
              "bayes_rep" ~ "Replication Bayes factor",
              "correspondence" ~ "Correspondence test",
            )
          ) %>% 
          arrange(metric)
        
        complete_cases <- repli_binary %>% 
          left_join(repli_outcomes_orig %>% select(report_id, score = repli_score_criteria_met), by = "report_id") %>% 
          rename_with(., function(x) str_remove_all(x, "repli_binary_"), .cols = everything()) %>% 
          drop_na() %>% 
          pivot_longer(cols = c(-report_id,-paper_id,-claim_id), names_to = "metric", values_to = "outcome") %>% 
          group_by(metric) %>% 
          dplyr::summarize(
            success = sum(outcome),
            tot = n(),
            Rate = glue("{100*round(success/tot, 2)}%")
          ) %>% 
          mutate(Counts = glue("{success} of {tot}")) %>% 
          select(metric, Counts, Rate) %>% 
          mutate(
            metric = case_match(
              metric,
              "analyst" ~ "Analyst interpretation",
              "score" ~ "Sig + pattern",
              "sum_p" ~ "Sum of p-values",
              "skep_p" ~ "Skeptical p-value",
              "orig_wthn" ~ "Orig. in rep. CI",
              "rep_wthn" ~ "Rep. in orig. CI",
              "wthn_pi" ~ "Rep. in prediction interval",
              "meta_success" ~ "Meta-analysis",
              "bma_result" ~ "Bayesian meta-analysis",
              "telescopes" ~ "Small telescopes",
              "bf_result" ~ "Bayes factor",
              "bayes_rep" ~ "Replication Bayes factor",
              "correspondence" ~ "Correspondence test",
            )
          ) %>% 
          arrange(metric)
        
        table_s12 <- bind_cols(all_cases, complete_cases %>% select(-metric))
        
        for (row in 1:nrow(table_s12)){
          for (col in 1:ncol(table_s12)){
            assign(paste0("table_s12_",row,"_",col),
                   table_s12[[row,col]])
          }
        }
      }
      
      # Table S13: Proportion of new versus secondary data replications and power to detect 75% of the original effect size by discipline
      {
        fields.order <- sort(c("Psychology and Health","Business","Sociology and Criminology",
                               "Economics and Finance","Political Science","Education"))
        
        table_s13 <- do.call(rbind,lapply(fields.order,function(field_selected) {
          #field_selected <- fields.order[2]
          
          table.data <- repli_outcomes_merged %>%
            filter(field==field_selected) %>%
            group_by(paper_id) %>%
            mutate(weight = 1/n())
          
          c1 <- paste0(format.round(
            100*sum(as.integer(table.data$repli_type=="new data")*table.data$weight)/
              sum(table.data$weight),1),"%")
          
          c2 <- paste0(format.round(
            100*sum(as.integer(table.data$repli_type=="secondary data")*table.data$weight)/
              sum(table.data$weight),1),"%")
          
          table.data <- repli_outcomes_merged %>%
            filter(field==field_selected,
                   !is.na(combined_power_75)) %>%
            group_by(paper_id) %>%
            mutate(weight = 1/n())
          c3 <- paste0(format.round(
            100*weighted.median(table.data$combined_power_75,table.data$weight),1),"%")
          
          table.data <- repli_outcomes_merged %>%
            filter(field==field_selected) %>%
            group_by(paper_id) %>%
            mutate(weight = 1/n())
          
          c4 <- weighted.median(table.data$orig_sample_size_value,table.data$weight)
          c5 <- weighted.median(table.data$repli_sample_size_value,table.data$weight)
          
          data.frame(c1,c2,c3,c4,c5)
        }))
        
        for (row in 1:nrow(table_s13)){
          for (col in 1:ncol(table_s13)){
            assign(paste0("table_s13_",row,"_",col),
                   table_s13[row,col])
          }}
      }
      
      # Table S14: Original and replication outcomes by statistical significance and pattern and by Pearson's r effect size for 6 subdisciplines across claims
      {
        score_c6 <- repli_outcomes_orig %>% 
          filter(!is_covid & repli_version_of_record) %>% 
          select(report_id, paper_id, score = repli_score_criteria_met) %>% 
          left_join(paper_metadata %>% select(paper_id, field = COS_pub_category), by = "paper_id") %>% 
          select(-paper_id) %>% 
          mutate(field = str_to_sentence(field)) %>% 
          group_by(field) %>% 
          dplyr::summarize(
            success = sum(score),
            tot = n()
          ) %>% 
          bind_rows(summarize_at(., vars(-field), sum)) %>% 
          replace_na(., list(field = "Total")) %>% 
          mutate(
            Counts = glue("{success} of {tot}"),
            Rate = glue("{100*round(success/tot, 2)}%")
          ) %>% 
          select(-success, -tot)
        
        cor_c6 <- repli_outcomes_orig %>%
          filter(!is_covid & repli_version_of_record) %>% 
          left_join(orig_outcomes %>% select(claim_id, orig_conv_r), by = "claim_id") %>% 
          mutate(orig_conv_r = abs(orig_conv_r)) %>% 
          mutate(repli_conv_r = ifelse(repli_pattern_criteria_met, abs(repli_conv_r), -1*abs(repli_conv_r))) %>% 
          select(report_id, paper_id, orig_conv_r, repli_conv_r) %>% 
          left_join(paper_metadata %>% select(paper_id, field = COS_pub_category), by = "paper_id") %>% 
          select(-paper_id) %>% 
          mutate(field = str_to_sentence(field)) %>% 
          pivot_longer(cols = c(orig_conv_r, repli_conv_r), names_to = "outcome", values_to = "value") %>% 
          group_by(field, outcome) %>% 
          dplyr::summarize(
            m = median(value, na.rm = T) %>% round(2),
            s = sd(value, na.rm = T) %>% round(2)
          ) %>% 
          mutate(t = glue("{m} ({s})")) %>% 
          ungroup() %>% 
          select(-m, -s) %>% 
          pivot_wider(names_from = outcome, values_from = t)
        
        table_s14 <- score_c6 %>% 
          left_join(cor_c6, by = "field") %>% 
          mutate(across(everything(), function(x) ifelse(is.na(x), "--", x)))
        
        for (row in 1:nrow(table_s14)){
          for (col in 1:ncol(table_s14)){
            assign(paste0("table_s14_",row,"_",col),
                   table_s14[[row,col]])
          }
        }
        
      }
      
      # Table S15: Original and replication outcomes by statistical significance and pattern and by Pearson's r effect size for 6 subdisciplines separated by new and secondary data across claims
      {
        tab_s15_left <- repli_outcomes %>%
          left_join(paper_metadata %>% select(paper_id, field = COS_pub_category), by = "paper_id") %>%
          rename(score = repli_score_criteria_met) %>%
          group_by(field, repli_type) %>%
          dplyr::summarize(
            success = sum(score),
            tot = n()
          ) %>%
          ungroup() %>%
          bind_rows(
            group_by(., repli_type) %>%
              dplyr::summarize(
                field = "Total",
                success = sum(success),
                tot = sum(tot)
              )
          ) %>%
          mutate(
            Counts = glue("{success} of {tot}"),
            Rate = glue("{100*round(success/tot, 2)}%")
          ) %>%
          select(-success, -tot) %>%
          mutate(field = str_to_sentence(field)) %>%
          arrange(repli_type, field)
        
        tab_s15_right_r <- repli_outcomes %>%
          mutate(repli_conv_r = ifelse(repli_pattern_criteria_met, abs(repli_conv_r), -1*abs(repli_conv_r))) %>%
          left_join(paper_metadata %>% select(paper_id, field = COS_pub_category), by = "paper_id") %>%
          group_by(field, repli_type) %>%
          dplyr::summarize(
            m = median(repli_conv_r, na.rm = T) %>% round(2) %>% as.character(),
            s = sd(repli_conv_r, na.rm = T) %>% round(2) %>% as.character()
          ) %>%
          mutate(s = ifelse(is.na(s), "--", s)) %>%
          mutate(r = glue("{m} ({s})")) %>%
          ungroup() %>%
          select(-m, -s) %>%
          mutate(field = str_to_sentence(field))
        
        tab_s15_right <- orig_outcomes %>%
          semi_join(
            repli_outcomes,
            by = "claim_id"
          ) %>%
          left_join(
            repli_outcomes %>%
              select(claim_id, repli_type),
            by = "claim_id"
          ) %>%
          mutate(orig_conv_r = abs(orig_conv_r)) %>%
          select(!field) %>%
          left_join(paper_metadata %>% select(paper_id, field = COS_pub_category), by = "paper_id") %>%
          group_by(field, repli_type) %>%
          dplyr::summarize(
            m = median(orig_conv_r, na.rm = T) %>% round(2) %>% as.character(),
            s = sd(orig_conv_r, na.rm = T) %>% round(2) %>% as.character()
          ) %>%
          mutate(s = ifelse(is.na(s), "--", s)) %>%
          mutate(o = glue("{m} ({s})")) %>%
          ungroup() %>%
          select(-m, -s) %>%
          mutate(field = str_to_sentence(field)) %>%
          left_join(tab_s15_right_r, by = c("field", "repli_type")) %>%
          arrange(repli_type, field)
        
        table_s15 <- tab_s15_left %>%
          left_join(tab_s15_right, by = c("field", "repli_type")) %>%
          mutate(
            across(
              c("o", "r"),
              function(x) ifelse(is.na(x), "--", x)
            )
          ) %>%
          select(-repli_type)
        
        for (row in 1:nrow(table_s15)){
          for (col in 1:ncol(table_s15)){
            assign(paste0("table_s15_",row,"_",col),
                   table_s15[[row,col]])
          }
        }
      }

      # Table S16: Original and replication outcomes by statistical significance and pattern and by Pearson's r effect size for 12 subdisciplines across claims
      {
        exp_fields <- paper_metadata %>% 
          select(paper_id, pub = publication_standard, field = COS_pub_expanded) %>% 
          mutate(
            field = case_when(
              str_detect(pub, "financ|Financ") ~ "finance",
              str_detect(pub, "organization|Organization") ~ "org. behavior",
              !str_detect(pub, "organization|Organization") & field == "marketing/org behavior"  ~ "marketing",
              !str_detect(pub, "financ|Financ") & field == "economics"  ~ "economics",
              .default = field
            )
          )
        
        score_c12 <- repli_outcomes_orig %>% 
          filter(!is_covid & repli_version_of_record) %>% 
          select(report_id, paper_id, score = repli_score_criteria_met) %>% 
          left_join(exp_fields , by = "paper_id") %>% 
          select(-paper_id) %>% 
          mutate(field = str_to_sentence(field)) %>% 
          group_by(field) %>% 
          dplyr::summarize(
            success = sum(score),
            tot = n()
          ) %>% 
          bind_rows(summarize_at(., vars(-field), sum)) %>% 
          replace_na(., list(field = "Total")) %>% 
          mutate(
            Counts = glue("{success} of {tot}"),
            Rate = glue("{100*round(success/tot, 2)}%")
          ) %>% 
          select(-success, -tot)
        
        cor_c12 <- repli_outcomes_orig %>%
          filter(!is_covid & repli_version_of_record) %>% 
          left_join(orig_outcomes %>% select(claim_id, orig_conv_r), by = "claim_id") %>% 
          mutate(orig_conv_r = abs(orig_conv_r)) %>% 
          mutate(repli_conv_r = ifelse(repli_pattern_criteria_met, abs(repli_conv_r), -1*abs(repli_conv_r))) %>% 
          select(report_id, paper_id, orig_conv_r, repli_conv_r) %>% 
          left_join(exp_fields , by = "paper_id") %>% 
          select(-paper_id) %>% 
          mutate(field = str_to_sentence(field)) %>% 
          pivot_longer(cols = c(orig_conv_r, repli_conv_r), names_to = "outcome", values_to = "value") %>% 
          group_by(field, outcome) %>% 
          dplyr::summarize(
            m = median(value, na.rm = T) %>% round(2),
            s = sd(value, na.rm = T) %>% round(2)
          ) %>% 
          mutate(t = glue("{m} ({s})")) %>% 
          ungroup() %>% 
          select(-m, -s) %>% 
          pivot_wider(names_from = outcome, values_from = t)
        
        table_s16 <- score_c12 %>% 
          left_join(cor_c12, by = "field") %>% 
          mutate(across(everything(), function(x) ifelse(is.na(x), "--", x))) 
        
        for (row in 1:nrow(table_s16)){
          for (col in 1:ncol(table_s16)){
            assign(paste0("table_s16_",row,"_",col),
                   table_s16[[row,col]])
          }
        }
      }
      
      # Table S17: Outcomes of individual data collections of the same original claim (3 cases)
      {
        table_s17 <- repli_outcomes_orig %>% 
          filter(!is_covid) %>% 
          filter(is_manylabs == "traditional") %>% 
          group_by(rr_id) %>% 
          arrange(desc(repli_sample_size_value)) %>% 
          dplyr::slice(1) %>% 
          ungroup() %>% 
          group_by(paper_id) %>% 
          arrange(paper_id, desc(manylabs_type)) %>% 
          ungroup() %>% 
          left_join(orig_outcomes %>% select(claim_id, orig_conv_r, orig_conv_r_lb, orig_conv_r_ub), by = "claim_id") %>% 
          mutate(across(where(is.numeric),function(x) round(x, 2))) %>% 
          mutate(original_effect = glue("{orig_conv_r} ({orig_conv_r_lb}, {orig_conv_r_ub})")) %>% 
          mutate(replication_effect = glue("{repli_conv_r} ({repli_conv_r_lb}, {repli_conv_r_ub})")) %>% 
          left_join(paper_metadata %>% select(paper_id, publication_standard), by = "paper_id") %>% 
          mutate(repli_sample_size_value = as.character(repli_sample_size_value)) %>% 
          mutate(repli_sample_size_value = ifelse(manylabs_type == "aggregation", "--", repli_sample_size_value)) %>% 
          select(paper_id, publication_standard, original_effect, rr_id, repli_sample_size_value, replication_effect) %>% 
          mutate(
            paper_id = case_match(
              paper_id,
              "G1Lr" ~ "Yang et al. (2013)",
              "Q1dl" ~ "King & Bryant (2017)",
              "YpZZ" ~ "Ku & Zaroff (2014)"
            )
          ) %>% 
          mutate(replication_effect = ifelse(rr_id == "999g", "-0.12 (-0.28, 0.04)", replication_effect))
        
        for (row in 1:nrow(table_s17)){
          for (col in 1:ncol(table_s17)){
            assign(paste0("table_s17_",row,"_",col),
                   table_s17[[row,col]])
          }
        }
      }
      
      # Table S18: Outcomes of individual data collections of the same claim using different protocols
      {
        table_s18 <- repli_outcomes_orig %>% 
          filter(!is_covid) %>% 
          filter(is_manylabs == "posthoc") %>% 
          group_by(rr_id) %>% 
          arrange(desc(repli_sample_size_value)) %>% 
          dplyr::slice(1) %>% 
          ungroup() %>% 
          group_by(paper_id) %>% 
          arrange(paper_id, desc(manylabs_type)) %>% 
          ungroup() %>% 
          left_join(orig_outcomes %>% select(claim_id, orig_conv_r, orig_conv_r_lb, orig_conv_r_ub), by = "claim_id") %>% 
          mutate(across(where(is.numeric),function(x) round(x, 2))) %>% 
          mutate(original_effect = glue("{orig_conv_r} ({orig_conv_r_lb}, {orig_conv_r_ub})")) %>% 
          mutate(replication_effect = glue("{repli_conv_r} ({repli_conv_r_lb}, {repli_conv_r_ub})")) %>% 
          left_join(paper_metadata %>% select(paper_id, publication_standard), by = "paper_id") %>% 
          mutate(repli_sample_size_value = as.character(repli_sample_size_value)) %>% 
          mutate(repli_sample_size_value = ifelse(manylabs_type == "aggregation", "--", repli_sample_size_value)) %>% 
          select(paper_id, publication_standard, original_effect, rr_id, repli_sample_size_value, replication_effect) %>% 
          mutate(replication_effect = ifelse(rr_id == "24316", "0.14 (0.03, 0.24)", replication_effect)) %>% 
          mutate(
            paper_id = case_match(
              paper_id,
              "8wZ0" ~ "Trémolière et al. (2012)",
              "AgO1" ~ "Rodriguez-Lara & Moreno-Garrido (2012)",
              "AvOr" ~ "Alves et al. (2018)",
              "Br0x" ~ "Bhattacharjee et al. (2017)",
              "EQxa" ~ "Pastötter et al. (2013)",
              "J7ek" ~ "Griffiths & Tenenbaum (2011)",
              "LyLd" ~ "Fox (2009)",
              "mJDj" ~ "Armon et al. (2013)",
              "plLK" ~ "McCarter et al. (2010)",
              "rjb" ~ "Luttrell et al. (2016)",
              "zlBL" ~ "Boehm et al. (2015)",
              "zlm2" ~ "Ohtsubo et al. (2014)"
              
            )
          )
        
        for (row in 1:nrow(table_s18)){
          for (col in 1:ncol(table_s18)){
            assign(paste0("table_s18_",row,"_",col),
                   table_s18[[row,col]])
          }
        }
      }
      
      # Table S19: Effects for hybrid replication attempts that included original and independent data
      {
        table_s19 <-
          repli_outcomes_orig %>%
          filter(!is_covid) %>%
          filter(repli_type == "original and secondary data") %>%
          mutate(full_id = select(., c(claim_id, rr_id)) %>% apply(1, function(x) str_c(x, collapse = "-"))) %>%
          bind_rows(
            repli_outcomes %>%
              filter(repli_type != "original and secondary data") %>%
              mutate(full_id = select(., c(claim_id, rr_id)) %>% apply(1, function(x) str_c(x, collapse = "-")))
          ) %>%
          group_by(full_id) %>%
          mutate(hybrid_check = paste0(repli_type, collapse = ", ")) %>%
          ungroup() %>%
          filter(hybrid_check != "new data" & hybrid_check != "new data, new data" & hybrid_check != "secondary data") %>%
          arrange(full_id) %>%
          left_join(orig_outcomes %>% select(claim_id, orig_conv_r, orig_conv_r_lb, orig_conv_r_ub), by = "claim_id") %>%
          mutate(across(where(is.numeric),function(x) round(x, 2))) %>%
          mutate(original_effect = glue("{orig_conv_r} ({orig_conv_r_lb}, {orig_conv_r_ub})")) %>%
          mutate(replication_effect = glue("{repli_conv_r} ({repli_conv_r_lb}, {repli_conv_r_ub})")) %>%
          left_join(paper_metadata %>% select(paper_id, publication_standard), by = "paper_id") %>%
          mutate(repli_sample_size_value = as.character(repli_sample_size_value)) %>%
          select(paper_id, publication_standard, claim_id, original_effect, rr_id, repli_type,
                 repli_sample_size_value, replication_effect, outcome = repli_score_criteria_met) %>%
          mutate(
            across(
              .cols = c(original_effect, replication_effect),
              .fns = function(x) ifelse(x == "NA (NA, NA)", "--", x)
            )
          ) %>%
          mutate(repli_type = select(., repli_type) %>% apply(1, function(x) ifelse(x == "secondary data", "Replication", "Hybrid"))) %>%
          mutate(outcome = ifelse(outcome, "Success", "Failed")) %>%
          mutate(repli_sample_size_value = ifelse(rr_id == "yk20", "713", repli_sample_size_value))
        
        for (row in 1:nrow(table_s19)){
          for (col in 1:ncol(table_s19)){
            assign(paste0("table_s19_",row,"_",col),
                   table_s19[[row,col]])
          }
        }
      }
    }
    
    # Attrition of replication attempts selected in Phase 1
    {
      n_papers_repl_team_ident_but_not_complete <- rr_type_key %>%
        filter(type == "replication") %>%
        semi_join(paper_status_tracking %>% filter(p1_delivery), by = "paper_id") %>%
        anti_join(repli_all_cases, by = "paper_id") %>%
        select(paper_id) %>%
        distinct() %>%
        nrow()
      
      n_papers_repli_never_started <- rr_type_key %>%
        filter(type == "replication") %>%
        semi_join(paper_status_tracking %>% filter(p1_delivery), by = "paper_id") %>%
        anti_join(repli_all_cases, by = "paper_id") %>%
        select(paper_id) %>%
        distinct() %>%
        anti_join(
          rr_internal_tracking %>%
            filter(field != "covid") %>%
            filter(str_detect(type, "Replication")),
          by = "paper_id"
        ) %>% nrow()
      
      p_papers_repli_never_started <- paste0(format.round(
        100*n_papers_repli_never_started/n_papers_repl_team_ident_but_not_complete,1
      ),"%")
      
      n_papers_repli_OSF_no_prereg <- rr_type_key %>%
        filter(type == "replication") %>%
        semi_join(paper_status_tracking %>% filter(p1_delivery), by = "paper_id") %>%
        anti_join(repli_all_cases, by = "paper_id") %>%
        select(paper_id) %>%
        distinct() %>%
        semi_join(
          rr_internal_tracking %>%
            filter(field != "covid") %>%
            filter(str_detect(type, "Replication")),
          by = "paper_id"
        ) %>%
        left_join(
          rr_internal_tracking %>%
            filter(field != "covid") %>%
            filter(str_detect(type, "Replication")),
          by = "paper_id"
        ) %>%
        mutate(
          registered = !outcome & results_available == "no" & (!is.na(registrations) | prereg_completion == "approve"),
          prereg = !outcome & results_available == "no" & is.na(registrations) & prereg_completion == "complete",
          part = !outcome & results_available == "no" & is.na(registrations) & prereg_completion %in% c("near-complete", "partial", "minimal"),
        ) %>%
        filter(is.na(registered) & is.na(prereg) & !part) %>%
        nrow()
        
      p_papers_repli_OSF_no_prereg <- paste0(format.round(
        100*n_papers_repli_OSF_no_prereg/n_papers_repl_team_ident_but_not_complete,1
      ),"%")
      
      n_papers_repli_prereg_started_not_finished <- rr_type_key %>%
        filter(type == "replication") %>%
        semi_join(paper_status_tracking %>% filter(p1_delivery), by = "paper_id") %>%
        anti_join(repli_all_cases, by = "paper_id") %>%
        select(paper_id) %>%
        distinct() %>%
        semi_join(
          rr_internal_tracking %>%
            filter(field != "covid") %>%
            filter(str_detect(type, "Replication")),
          by = "paper_id"
        ) %>%
        left_join(
          rr_internal_tracking %>%
            filter(field != "covid") %>%
            filter(str_detect(type, "Replication")),
          by = "paper_id"
        ) %>%
        mutate(
          registered = !outcome & results_available == "no" & (!is.na(registrations) | prereg_completion == "approve"),
          prereg = !outcome & results_available == "no" & is.na(registrations) & prereg_completion == "complete",
          part = !outcome & results_available == "no" & is.na(registrations) & prereg_completion %in% c("near-complete", "partial", "minimal"),
        ) %>%
        filter(part) %>%
        nrow()
      
      p_papers_repli_prereg_started_not_finished <- paste0(format.round(
        100*n_papers_repli_prereg_started_not_finished/n_papers_repl_team_ident_but_not_complete,1
      ),"%")
      
      n_papers_repli_completed_prereg <- rr_type_key %>%
        filter(type == "replication") %>%
        semi_join(paper_status_tracking %>% filter(p1_delivery), by = "paper_id") %>%
        anti_join(repli_all_cases, by = "paper_id") %>%
        select(paper_id) %>%
        distinct() %>%
        semi_join(
          rr_internal_tracking %>%
            filter(field != "covid") %>%
            filter(str_detect(type, "Replication")),
          by = "paper_id"
        ) %>%
        left_join(
          rr_internal_tracking %>%
            filter(field != "covid") %>%
            filter(str_detect(type, "Replication")),
          by = "paper_id"
        ) %>%
        mutate(
          registered = !outcome & results_available == "no" & (!is.na(registrations) | prereg_completion == "approve"),
          prereg = !outcome & results_available == "no" & is.na(registrations) & prereg_completion == "complete",
          part = !outcome & results_available == "no" & is.na(registrations) & prereg_completion %in% c("near-complete", "partial", "minimal"),
        ) %>%
        filter(prereg & !registered) %>%
        nrow()
      
      p_papers_repli_completed_prereg <- paste0(format.round(
        100*n_papers_repli_completed_prereg/n_papers_repl_team_ident_but_not_complete,1
      ),"%")
      
      n_papers_repli_registered_study <- rr_type_key %>%
        filter(type == "replication") %>%
        semi_join(paper_status_tracking %>% filter(p1_delivery), by = "paper_id") %>%
        anti_join(repli_all_cases, by = "paper_id") %>%
        select(paper_id) %>%
        distinct() %>%
        semi_join(
          rr_internal_tracking %>%
            filter(field != "covid") %>%
            filter(str_detect(type, "Replication")),
          by = "paper_id"
        ) %>%
        left_join(
          rr_internal_tracking %>%
            filter(field != "covid") %>%
            filter(str_detect(type, "Replication")),
          by = "paper_id"
        ) %>%
        mutate(
          registered = !outcome & results_available == "no" & (!is.na(registrations) | prereg_completion == "approve"),
          prereg = !outcome & results_available == "no" & is.na(registrations) & prereg_completion == "complete",
          part = !outcome & results_available == "no" & is.na(registrations) & prereg_completion %in% c("near-complete", "partial", "minimal"),
        ) %>%
        filter(prereg & !registered) %>%
        nrow()
      
      p_papers_repli_registered_study <- paste0(format.round(
        100*n_papers_repli_registered_study/n_papers_repl_team_ident_but_not_complete,1
      ),"%")
    }
    
    # Non-random selection and no attrition of replications selected in Phase 2
    {
      n_papers_p2_repli_completed_nd <- 
        repli_outcomes %>%
        filter(type_internal == "p2") %>%
        filter(repli_type == "new data") %>%
        nrow()
      
      n_papers_p2_repli_completed_sd <- 
        repli_outcomes %>%
        filter(type_internal == "p2") %>%
        filter(repli_type == "secondary data") %>%
        nrow()
    }
    
    # Excluded cases
    {
      id_excluded_case_orig_negative <-  repli_case_exclusions %>%
        left_join(repli_all_cases %>% select(rr_id, paper_id) %>% distinct(), by = "rr_id") %>%
        filter(exclude_reason == "Non-significant original findings") %>%
        pull(paper_id) %>%
        unique()
      
      id_excluded_sample_size_too_small <- repli_case_exclusions %>%
        left_join(repli_all_cases %>% select(rr_id, paper_id) %>% distinct(), by = "rr_id") %>%
        filter(exclude_reason == "Incomplete or insufficient data collection") %>%
        pull(paper_id) %>%
        unique() %>%
        str_c(., collapse = ", ")
      
      n_excluded_power_all_approach_nd <- repli_case_exclusions %>%
        left_join(repli_all_cases %>% select(rr_id, paper_id) %>% distinct(), by = "rr_id") %>%
        filter(exclude_reason == "Power/sample size issue") %>%
        left_join(repli_all_cases %>% select(rr_id, rr_type_internal) %>% distinct(), by = "rr_id") %>%
        filter(rr_type_internal == "Direct Replication") %>%
        nrow()
      
      id_excluded_power_all_approach_nd <- repli_case_exclusions %>%
        left_join(repli_all_cases %>% select(rr_id, paper_id) %>% distinct(), by = "rr_id") %>%
        filter(exclude_reason == "Power/sample size issue") %>%
        left_join(repli_all_cases %>% select(rr_id, rr_type_internal) %>% distinct(), by = "rr_id") %>%
        filter(rr_type_internal == "Direct Replication") %>%
        pull(paper_id) %>%
        unique() %>%
        str_c(., collapse = ", ")
      
      n_excluded_power_all_approach_sd <-  repli_case_exclusions %>%
        left_join(repli_all_cases %>% select(rr_id, paper_id) %>% distinct(), by = "rr_id") %>%
        filter(exclude_reason == "Power/sample size issue") %>%
        left_join(repli_all_cases %>% select(rr_id, rr_type_internal) %>% distinct(), by = "rr_id") %>%
        filter(rr_type_internal == "Data Analytic Replication") %>%
        nrow()
      
      id_excluded_power_all_approach_sd <- repli_case_exclusions %>%
        left_join(repli_all_cases %>% select(rr_id, paper_id) %>% distinct(), by = "rr_id") %>%
        filter(exclude_reason == "Power/sample size issue") %>%
        left_join(repli_all_cases %>% select(rr_id, rr_type_internal) %>% distinct(), by = "rr_id") %>%
        filter(rr_type_internal == "Data Analytic Replication") %>%
        pull(paper_id) %>%
        unique() %>%
        str_c(., collapse = ", ")
      
      n_claims_excluded <- repli_case_exclusions %>%
        left_join(repli_all_cases %>% select(rr_id, paper_id) %>% distinct(), by = "rr_id") %>%
        nrow()
      
      n_papers_excluded <- repli_case_exclusions %>%
        left_join(repli_all_cases %>% select(rr_id, paper_id) %>% distinct(), by = "rr_id") %>%
        pull(paper_id) %>%
        unique() %>%
        length()
      
      n_excluded_claims_nonsig <- repli_case_exclusions %>%
        filter(exclude_reason != "Non-significant original findings") %>%
        nrow()
      
      n_excluded_claims_nonsig_success <- repli_case_exclusions %>%
        filter(exclude_reason != "Non-significant original findings") %>%
        left_join(repli_all_cases %>% select(report_id = report_id, score = rr_repl_exact_replicated_reference), by = "report_id") %>%
        filter(score) %>%
        nrow()
    }
    
    # Sample and Data
    {
      n_claims_potential_repli <- paper_status_tracking %>%
        filter(pool) %>%
        nrow()
      
      n_claims_eligible_nonrand <- paper_status_tracking %>%
        filter(bushel) %>%
        nrow()
    }
    
    # Sample Size Planning 
    {
      # 50%/90 power
        # New data replications conducted during Phase 1
        p_claim_nd_attempt_50_90_power_p1 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 1") %>%
          filter(repli_type == "new data") %>%
          dplyr::summarize(m = 100*mean(combined_power_50 >= .9, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
      
        median_claim_nd_attempt_50_90_power_p1 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 1") %>%
          filter(repli_type == "new data") %>%
          dplyr::summarize(m = 100*median(combined_power_50 >= .9, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        p_claim_nd_attempt_50_90_power_achieved_p1 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 1") %>%
          filter(repli_type == "new data") %>%
          dplyr::summarize(m = 100*mean(combined_power_50, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        median_claim_nd_attempt_50_90_power_achieved_p1 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 1") %>%
          filter(repli_type == "new data") %>%
          dplyr::summarize(m = 100*median(combined_power_50, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        # New data replications conducted during Phase 2
        p_claim_nd_attempt_50_90_power_p2 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 2") %>%
          filter(repli_type == "new data") %>%
          dplyr::summarize(m = 100*mean(combined_power_50 >= .9, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        median_claim_nd_attempt_50_90_power_p2 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 2") %>%
          filter(repli_type == "new data") %>%
          dplyr::summarize(m = 100*median(combined_power_50 >= .9, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        p_claim_nd_attempt_50_90_power_achieved_p2 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 2") %>%
          filter(repli_type == "new data") %>%
          dplyr::summarize(m = 100*mean(combined_power_50, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        median_claim_nd_attempt_50_90_power_achieved_p2 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 2") %>%
          filter(repli_type == "new data") %>%
          dplyr::summarize(m = 100*median(combined_power_50, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        # Secondary data replications conducted during Phase 1
        p_claim_sd_attempt_50_90_power_p1 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 1") %>%
          filter(repli_type == "secondary data") %>%
          dplyr::summarize(m = 100*mean(combined_power_50 >= .9, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        median_claim_sd_attempt_50_90_power_p1 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 1") %>%
          filter(repli_type == "secondary data") %>%
          dplyr::summarize(m = 100*median(combined_power_50 >= .9, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        p_claim_sd_attempt_50_90_power_achieved_p1 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 1") %>%
          filter(repli_type == "secondary data") %>%
          dplyr::summarize(m = 100*mean(combined_power_50, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        median_claim_sd_attempt_50_90_power_achieved_p1 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 1") %>%
          filter(repli_type == "secondary data") %>%
          dplyr::summarize(m = 100*median(combined_power_50, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        # Secondary data replications conducted during Phase 2
        p_claim_sd_attempt_50_90_power_p2 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 2") %>%
          filter(repli_type == "secondary data") %>%
          dplyr::summarize(m = 100*mean(combined_power_50 >= .9, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        median_claim_sd_attempt_50_90_power_p2 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 2") %>%
          filter(repli_type == "secondary data") %>%
          dplyr::summarize(m = 100*median(combined_power_50 >= .9, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        p_claim_sd_attempt_50_90_power_achieved_p2 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 2") %>%
          filter(repli_type == "secondary data") %>%
          dplyr::summarize(m = 100*mean(combined_power_50, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        median_claim_sd_attempt_50_90_power_achieved_p2 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 2") %>%
          filter(repli_type == "secondary data") %>%
          dplyr::summarize(m = 100*median(combined_power_50, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
      
      # 75%/90 power
        # New data replications conducted during Phase 1
        p_claim_nd_attempt_75_90_power_p1 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 1") %>%
          filter(repli_type == "new data") %>%
          dplyr::summarize(m = 100*mean(combined_power_75 >= .9, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        median_claim_nd_attempt_75_90_power_p1 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 1") %>%
          filter(repli_type == "new data") %>%
          dplyr::summarize(m = 100*median(combined_power_75 >= .9, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        p_claim_nd_attempt_75_90_power_achieved_p1 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 1") %>%
          filter(repli_type == "new data") %>%
          dplyr::summarize(m = 100*mean(combined_power_75, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        median_claim_nd_attempt_75_90_power_achieved_p1 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 1") %>%
          filter(repli_type == "new data") %>%
          dplyr::summarize(m = 100*median(combined_power_75, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        # New data replications conducted during Phase 2
        p_claim_nd_attempt_75_90_power_p2 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 2") %>%
          filter(repli_type == "new data") %>%
          dplyr::summarize(m = 100*mean(combined_power_75 >= .9, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        median_claim_nd_attempt_75_90_power_p2 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 2") %>%
          filter(repli_type == "new data") %>%
          dplyr::summarize(m = 100*median(combined_power_75 >= .9, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        p_claim_nd_attempt_75_90_power_achieved_p2 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 2") %>%
          filter(repli_type == "new data") %>%
          dplyr::summarize(m = 100*mean(combined_power_75, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        median_claim_nd_attempt_75_90_power_achieved_p2 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 2") %>%
          filter(repli_type == "new data") %>%
          dplyr::summarize(m = 100*median(combined_power_75, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        # Secondary data replications conducted during Phase 1
        p_claim_sd_attempt_75_90_power_p1 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 1") %>%
          filter(repli_type == "secondary data") %>%
          dplyr::summarize(m = 100*mean(combined_power_75 >= .9, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        median_claim_sd_attempt_75_90_power_p1 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 1") %>%
          filter(repli_type == "secondary data") %>%
          dplyr::summarize(m = 100*median(combined_power_75 >= .9, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        p_claim_sd_attempt_75_90_power_achieved_p1 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 1") %>%
          filter(repli_type == "secondary data") %>%
          dplyr::summarize(m = 100*mean(combined_power_75, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        median_claim_sd_attempt_75_90_power_achieved_p1 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 1") %>%
          filter(repli_type == "secondary data") %>%
          dplyr::summarize(m = 100*median(combined_power_75, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        # Secondary data replications conducted during Phase 2
        p_claim_sd_attempt_75_90_power_p2 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 2") %>%
          filter(repli_type == "secondary data") %>%
          dplyr::summarize(m = 100*mean(combined_power_75 >= .9, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        median_claim_sd_attempt_75_90_power_p2 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 2") %>%
          filter(repli_type == "secondary data") %>%
          dplyr::summarize(m = 100*median(combined_power_75 >= .9, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        p_claim_sd_attempt_75_90_power_achieved_p2 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 2") %>%
          filter(repli_type == "secondary data") %>%
          dplyr::summarize(m = 100*mean(combined_power_75, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
        
        median_claim_sd_attempt_75_90_power_achieved_p2 <- repli_outcomes %>%
          filter(repli_start_phase == "phase 2") %>%
          filter(repli_type == "secondary data") %>%
          dplyr::summarize(m = 100*median(combined_power_75, na.rm = T)) %>%
          pull(m) %>% format.round(1) %>%  paste0("%")
      
        
      # 50/100
        median_claim_sd_attempt_50_100_power_p1 <- paste0(format.round(100*(
          repli_outcomes %>%
            filter(repli_type == "secondary data") %>%
            filter(repli_start_phase == "phase 1") %>%
            drop_na(combined_power_100) %>%
            count(combined_power_100 >= .5) %>%
            mutate(prop = n/sum(n)) %>%
            filter(`combined_power_100 >= 0.5`) %>%
            pull(prop)
        ),1),"%")
        
        p_claim_sd_attempt_50_100_power_p1 <- paste0(format.round(100*(
          repli_outcomes %>%
          filter(repli_start_phase == "phase 1") %>%
          filter(repli_type == "secondary data") %>%
          dplyr::summarize(m = mean(combined_power_100 >= .5, na.rm = T)) %>%
          pull(m)
        ),1),"%")
        
          
        
        p_claim_sd_attempt_50_100_power_achieved_p1 <- paste0(format.round(100*(
          repli_outcomes_orig %>%
            filter(repli_type == "secondary data") %>%
            filter(repli_start_phase == "phase 1") %>%
            pull(combined_power_100) %>%
            mean(na.rm = T)
        ),1),"%")
        
        median_claim_sd_attempt_50_100_power_achieved_p1 <- paste0(format.round(100*(
          repli_outcomes_orig %>%
            filter(repli_type == "secondary data") %>%
            filter(repli_start_phase == "phase 1") %>%
            pull(combined_power_100) %>%
            median(na.rm = T)
        ),1),"%")
        
      median_claim_sd_attempt_50_100_power_p2 <- paste0(format.round(100*(
        repli_outcomes %>%
          filter(repli_type == "secondary data") %>%
          filter(repli_start_phase == "phase 2") %>%
          drop_na(combined_power_100) %>%
          count(combined_power_100 >= .5) %>%
          mutate(prop = n/sum(n)) %>%
          filter(`combined_power_100 >= 0.5`) %>%
          pull(prop)
      ),1),"%")

      p_claim_sd_attempt_50_100_power_achieved_p2 <- paste0(format.round(100*(
        repli_outcomes_orig %>%
          filter(repli_type == "secondary data") %>%
          filter(repli_start_phase == "phase 2") %>%
          pull(combined_power_100) %>%
          mean(na.rm = T)
      ),1),"%")

      median_claim_sd_attempt_50_100_power_achieved_p2 <- paste0(format.round(100*(
        repli_outcomes_orig %>%
          filter(repli_type == "secondary data") %>%
          filter(repli_start_phase == "phase 2") %>%
          pull(combined_power_100) %>%
          median(na.rm = T)
      ),1),"%")
    }
    
    # SER Method for estimating power versus traditional power analysis
    {
      n_claims_claims_ser_power <- 
        orig_outcomes %>%
        semi_join(repli_outcomes %>% filter(!is_covid & repli_version_of_record), by = c("claim_id" = "claim_id")) %>%
        filter(str_detect(orig_stat_type_for_repli, "ser")) %>%
        nrow()
      
      from_ser <- claims_ser_power %>% 
        rename(r_threshold = rr_sample_size_50_for_100, r_s1 = rr_sample_size_90_for_50, r_s2 = rr_sample_size_90_for_50) %>% 
        left_join(
          orig_outcomes %>% 
            select(claim_id, ser_threshold = orig_sample_size_50_for_100, 
                   ser_s1 = orig_sample_size_90_for_50, ser_s2 = orig_sample_size_90_for_50),
          by = "claim_id"
        ) %>% 
        pivot_longer(cols = -claim_id, names_to = "category", values_to = "value") %>% 
        separate(category, into = c("type", "category")) %>% 
        pivot_wider(names_from = type, values_from = value) %>% 
        drop_na()
      
      from_traditional <- claims_traditional_power %>% 
        drop_na(rr_sample_size_90_for_50) %>% 
        select(claim_id = claim_id, ser_threshold = rr_sample_size_50_for_100, 
               ser_s1 = rr_sample_size_90_for_50, ser_s2 = rr_sample_size_90_for_50) %>% 
        left_join(
          orig_outcomes %>% 
            select(claim_id, r_threshold = orig_sample_size_50_for_100, 
                   r_s1 = orig_sample_size_90_for_50, r_s2 = orig_sample_size_90_for_50),
          by = "claim_id"
        ) %>% 
        pivot_longer(cols = -claim_id, names_to = "category", values_to = "value") %>% 
        separate(category, into = c("type", "category")) %>% 
        pivot_wider(names_from = type, values_from = value)
      
      n_claims_SER_max_15000 <- 
        from_ser %>%
        bind_rows(from_traditional) %>%
        filter(r < 15000 & ser < 15000) %>% nrow()
      
      n_claims_SER_max_5000 <- from_ser %>%
        bind_rows(from_traditional) %>%
        filter(r < 5000 & ser < 5000) %>% nrow()
      
    }
    
    # Data collection & analysis
    {
      project_duration_p1_nd_mean <- 
        rr_project_dates %>%
        mutate(
          completed = case_when(
            completed > as.Date("2023-03-31") ~ as.Date("2023-03-31"),
            TRUE ~ completed
          )
        ) %>%
        semi_join(repli_outcomes %>% 
                    filter(!is_covid & (manylabs_type != "aggregation" | is.na(manylabs_type))), by = "rr_id") %>%
        mutate(days = completed - started) %>%
        group_by(type) %>%
        dplyr::summarize(
          m = mean(days, na.rm = T) %>% round(),
          s = sd(days, na.rm = T) %>% round()
        ) %>%
        mutate(m = as.integer(m)) %>%
        filter(type == "new_p1") %>%
        pull(m)
      
      project_duration_p1_nd_SD <- 
        rr_project_dates %>%
        mutate(
          completed = case_when(
            completed > as.Date("2023-03-31") ~ as.Date("2023-03-31"),
            TRUE ~ completed
          )
        ) %>%
        semi_join(repli_outcomes %>% 
                    filter(!is_covid & (manylabs_type != "aggregation" | is.na(manylabs_type))), by = "rr_id") %>%
        mutate(days = completed - started) %>%
        group_by(type) %>%
        dplyr::summarize(
          m = mean(days, na.rm = T) %>% round(),
          s = sd(days, na.rm = T) %>% round()
        ) %>%
        mutate(m = as.integer(m)) %>%
        filter(type == "new_p1") %>%
        pull(s)
    
    
      project_duration_p1_sd_mean <- 
        rr_project_dates %>%
        mutate(days = completed - started) %>%
        group_by(type) %>%
        dplyr::summarize(
          m = mean(days, na.rm = T) %>% round(),
          s = sd(days, na.rm = T) %>% round()
        ) %>%
        mutate(m = as.integer(m)) %>%
        filter(type == "secondary_p1") %>%
        pull(m)
      
      project_duration_p1_sd_SD <- 
        rr_project_dates %>%
        mutate(
          completed = case_when(
            completed > as.Date("2023-03-31") ~ as.Date("2023-03-31"),
            TRUE ~ completed
          )
        ) %>%
        semi_join(repli_outcomes %>% 
                    filter(!is_covid & (manylabs_type != "aggregation" | is.na(manylabs_type))), by = "rr_id") %>%
        mutate(days = completed - started) %>%
        group_by(type) %>%
        dplyr::summarize(
          m = mean(days, na.rm = T) %>% round(),
          s = sd(days, na.rm = T) %>% round()
        ) %>%
        mutate(m = as.integer(m)) %>%
        filter(type == "secondary_p1") %>%
        pull(s)
      
      project_duration_p2_nd_mean <- rr_project_dates %>%
        mutate(days = completed - started) %>%
        group_by(type) %>%
        dplyr::summarize(
          m = mean(days, na.rm = T) %>% round(),
          s = sd(days, na.rm = T) %>% round()
        ) %>%
        mutate(m = as.integer(m)) %>%
        filter(type == "new_p2") %>%
        pull(m)
      
      project_duration_p2_nd_SD <- 
        rr_project_dates %>%
        mutate(
          completed = case_when(
            completed > as.Date("2023-03-31") ~ as.Date("2023-03-31"),
            TRUE ~ completed
          )
        ) %>%
        semi_join(repli_outcomes %>% 
                    filter(!is_covid & (manylabs_type != "aggregation" | is.na(manylabs_type))), by = "rr_id") %>%
        mutate(days = completed - started) %>%
        group_by(type) %>%
        dplyr::summarize(
          m = mean(days, na.rm = T) %>% round(),
          s = sd(days, na.rm = T) %>% round()
        ) %>%
        mutate(m = as.integer(m)) %>%
        filter(type == "new_p2") %>%
        pull(s)
      
      project_duration_p2_sd_mean <- 
        rr_project_dates %>%
        mutate(
          completed = case_when(
            completed > as.Date("2023-03-31") ~ as.Date("2023-03-31"),
            TRUE ~ completed
          )
        ) %>%
        semi_join(repli_outcomes %>% 
                    filter(!is_covid & (manylabs_type != "aggregation" | is.na(manylabs_type))), by = "rr_id") %>%
        mutate(days = completed - started) %>%
        group_by(type) %>%
        dplyr::summarize(
          m = mean(days, na.rm = T) %>% round(),
          s = sd(days, na.rm = T) %>% round()
        ) %>%
        mutate(m = as.integer(m)) %>%
        filter(type == "secondary_p2") %>%
        pull(m)
      
      project_duration_p2_sd_SD <- 
        rr_project_dates %>%
        mutate(
          completed = case_when(
            completed > as.Date("2023-03-31") ~ as.Date("2023-03-31"),
            TRUE ~ completed
          )
        ) %>%
        semi_join(repli_outcomes %>% 
                    filter(!is_covid & (manylabs_type != "aggregation" | is.na(manylabs_type))), by = "rr_id") %>%
        mutate(days = completed - started) %>%
        group_by(type) %>%
        dplyr::summarize(
          m = mean(days, na.rm = T) %>% round(),
          s = sd(days, na.rm = T) %>% round()
        ) %>%
        mutate(m = as.integer(m)) %>%
        filter(type == "secondary_p2") %>%
        pull(s)
    }
    
    # Replications completed by year in comparison with the sampling frame
    {
      p_rep_2010_initial <- 
        paste0(format.round(100*table_s8_n$'2010'[1]/table_1_n$Total[1],1),"% [n = ",table_s8_n$'2010'[1],"/",table_s8_n$Total[1],"]")
      p_rep_2010_final <- 
        paste0(format.round(100*table_s8_n$'2010'[8]/table_1_n$Total[8],1),"% [n = ",table_s8_n$'2010'[8],"/",table_s8_n$Total[8],"]")
      
      p_rep_2011_initial <- 
        paste0(format.round(100*table_s8_n$'2011'[1]/table_1_n$Total[1],1),"% [n = ",table_s8_n$'2011'[1],"/",table_s8_n$Total[1],"]")
      p_rep_2011_final <- 
        paste0(format.round(100*table_s8_n$'2011'[8]/table_1_n$Total[8],1),"% [n = ",table_s8_n$'2011'[8],"/",table_s8_n$Total[8],"]")
      
      p_rep_2015_initial <- 
        paste0(format.round(100*table_s8_n$'2015'[1]/table_1_n$Total[1],1),"% [n = ",table_s8_n$'2015'[1],"/",table_s8_n$Total[1],"]")
      p_rep_2015_final <- 
        paste0(format.round(100*table_s8_n$'2015'[8]/table_1_n$Total[8],1),"% [n = ",table_s8_n$'2015'[8],"/",table_s8_n$Total[8],"]")
      
      p_max_expectation_repli_completed_except_2015 <- paste0(format.round(100*max(
        abs(table_s8_p$"2009"[6]-.1),
        abs(table_s8_p$"2010"[6]-.1),
        abs(table_s8_p$"2011"[6]-.1),
        abs(table_s8_p$"2012"[6]-.1),
        abs(table_s8_p$"2013"[6]-.1),
        abs(table_s8_p$"2014"[6]-.1),
        abs(table_s8_p$"2016"[6]-.1),
        abs(table_s8_p$"2017"[6]-.1),
        abs(table_s8_p$"2018"[6]-.1)
      ),1),"%")
      p_repli_completed_2015 <- paste0(format.round(100*
                                                      abs(table_s8_p$"2015"[6]),1),"%")
    }
    
    # Papers from Phase 1 for which a replication was not attempted 
    {
      n_replis_not_matched <- papers_not_sourced %>% nrow()
      
      p_replis_not_matched <- paste0(format.round(100*
                                                    (papers_not_sourced %>% nrow()) 
                                                  / (paper_status_tracking %>% filter(RR) %>% nrow()),
                                                  1),"%")
      
      n_replis_not_matched_plausible <- papers_not_sourced %>%
        count(dr_plausible) %>%
        filter(dr_plausible == "yes") %>%
        pull(n)
      
      p_replis_not_matched_plausible <- paste0(format.round(
        100*
          papers_not_sourced %>%
          count(dr_plausible) %>%
          mutate(prop = n/sum(n)) %>%
          filter(dr_plausible == "yes") %>%
          pull(prop),
        1),"%")
      
      n_replis_not_matched_implausible_secondary <- papers_not_sourced %>%
        filter(dr_reason == "secondary") %>%
        nrow()
      
      p_replis_not_matched_implausible_secondary <- paste0(format.round(
        100*
          papers_not_sourced %>%
          count(dr_reason) %>%
          mutate(prop = n/sum(n)) %>%
          filter(dr_reason == "secondary") %>%
          pull(prop),
        1),"%")
      
      n_replis_not_matched_implausible_sens_pop <- papers_not_sourced %>%
        count(dr_reason) %>%
        filter(dr_reason == "sensitive population") %>%
        pull(n)
      
      p_replis_not_matched_implausible_sens_pop <- paste0(format.round(
        100*
          papers_not_sourced %>%
          count(dr_reason) %>%
          mutate(prop = n/sum(n)) %>%
          filter(dr_reason == "sensitive population") %>%
          pull(prop),
        1),"%")
      
      n_replis_not_matched_implausible_long <- papers_not_sourced %>%
        count(dr_reason) %>%
        filter(dr_reason == "longitudinal") %>%
        pull(n)
        
      p_replis_not_matched_implausible_long <- paste0(format.round(
        100*
          papers_not_sourced %>%
          count(dr_reason) %>%
          mutate(prop = n/sum(n)) %>%
          filter(dr_reason == "longitudinal") %>%
          pull(prop),
        1),"%")
      
      n_replis_not_matched_implausible_unique <- papers_not_sourced %>%
        count(dr_reason) %>%
        filter(dr_reason == "unique event or program") %>%
        pull(n)
        
      p_replis_not_matched_implausible_unique <- paste0(format.round(
        100*
          papers_not_sourced %>%
          count(dr_reason) %>%
          mutate(prop = n/sum(n)) %>%
          filter(dr_reason == "unique event or program") %>%
          pull(prop),
        1),"%")
      
      n_replis_not_matched_2nd_plausible <- papers_not_sourced %>%
        count(dar_plausible) %>%
        filter(dar_plausible == "yes") %>%
        pull(n)
      
      p_replis_not_matched_2nd_plausible <- paste0(format.round(
        100*
          papers_not_sourced %>%
          count(dar_plausible) %>%
          mutate(prop = n/sum(n)) %>%
          filter(dar_plausible == "yes") %>%
          pull(prop),
        1),"%")
      
      n_replis_not_matched_2nd_implausible_nd <- papers_not_sourced %>%
        count(dar_reason) %>%
        filter(dar_reason == "own data collection") %>%
        pull(n)
      
      p_replis_not_matched_2nd_implausible_nd <- paste0(format.round(
        100*
          papers_not_sourced %>%
          count(dar_reason) %>%
          mutate(prop = n/sum(n)) %>%
          filter(dar_reason == "own data collection") %>%
          pull(prop),
        1),"%")
      
      n_replis_not_matched_2nd_implausible_unique <- papers_not_sourced %>%
        count(dar_reason) %>%
        #mutate(prop = n/sum(n)) %>%
        filter(dar_reason %in% c("archival data", "historical event", "unique data", "unique event or program")) %>%
        pull(n) %>%
        sum()
        
      p_replis_not_matched_2nd_implausible_unique <- paste0(format.round(
        100*
          papers_not_sourced %>%
          count(dar_reason) %>%
          mutate(prop = n/sum(n)) %>%
          filter(dar_reason %in% c("archival data", "historical event", "unique data", "unique event or program")) %>%
          pull(prop) %>%
          sum(),
        1),"%")
      
      n_replis_not_matched_2nd_implausible_insuf_data <- papers_not_sourced %>%
        count(dar_reason) %>%
        #mutate(prop = n/sum(n)) %>%
        filter(dar_reason == "insufficient new cases") %>%
        pull(n)
      
      p_replis_not_matched_2nd_implausible_insuf_data <- paste0(format.round(
        100*
          papers_not_sourced %>%
          count(dar_reason) %>%
          mutate(prop = n/sum(n)) %>%
          filter(dar_reason == "insufficient new cases") %>%
          pull(prop),
        1),"%")
      
      n_replis_not_matched_2nd_implausible_admin <- papers_not_sourced %>%
        count(dar_reason) %>%
        #mutate(prop = n/sum(n)) %>%
        filter(dar_reason == "administrative data") %>%
        pull(n)
        
      p_replis_not_matched_2nd_implausible_admin <- paste0(format.round(
        100*
          papers_not_sourced %>%
          count(dar_reason) %>%
          mutate(prop = n/sum(n)) %>%
          filter(dar_reason == "administrative data") %>%
          pull(prop),
        1),"%")
      
      n_replis_not_matched_2nd_implausible_long <- papers_not_sourced %>%
        count(dar_reason) %>%
        filter(dar_reason == "panel") %>%
        pull(n)
        
      p_replis_not_matched_2nd_implausible_long <- paste0(format.round(
        100*
          papers_not_sourced %>%
          count(dar_reason) %>%
          mutate(prop = n/sum(n)) %>%
          filter(dar_reason == "panel") %>%
          pull(prop),
        1),"%")
      
      n_replis_not_matched_plausible_p1 <- papers_not_sourced %>%
        count(dr_plausible == "yes" | dar_plausible == "yes") %>%
        filter(`dr_plausible == "yes" | dar_plausible == "yes"`) %>%
        pull(n)
        
      p_replis_not_matched_plausible_p1 <- paste0(format.round(
        100*
          papers_not_sourced %>%
          count(dr_plausible == "yes" | dar_plausible == "yes") %>%
          mutate(prop = n/sum(n)) %>%
          filter(`dr_plausible == "yes" | dar_plausible == "yes"`) %>%
          pull(prop),
        1),"%")
      
      n_replis_not_matched_implausible <- papers_not_sourced %>%
        count(dr_plausible == "yes" | dar_plausible == "yes") %>%
        filter(!(`dr_plausible == "yes" | dar_plausible == "yes"`)) %>%
        pull(n)
        
      p_replis_not_matched_implausible <- paste0(format.round(
        100*
          papers_not_sourced %>%
          count(dr_plausible == "yes" | dar_plausible == "yes") %>%
          mutate(prop = n/sum(n)) %>%
          filter(!(`dr_plausible == "yes" | dar_plausible == "yes"`)) %>%
          pull(prop),
        1),"%")

    }
    
    # Replication success rates across binary assessments
    {
      n_repli_binary_all_outcome_success_min <-
        repli_binary_orig %>%
        left_join(repli_outcomes_orig %>% select(report_id, score = repli_score_criteria_met), by = "report_id") %>%
        pivot_longer(cols = -report_id, names_to = "measure", values_to = "outcome") %>%
        drop_na() %>%
        group_by(measure) %>%
        dplyr::summarize(n = n()) %>%
        pull(n) %>%
        min()
      
      n_repli_binary_all_outcome_success_max <-
        repli_binary_orig %>%
        left_join(repli_outcomes_orig %>% select(report_id, score = repli_score_criteria_met), by = "report_id") %>%
        pivot_longer(cols = -report_id, names_to = "measure", values_to = "outcome") %>%
        drop_na() %>%
        group_by(measure) %>%
        dplyr::summarize(n = n()) %>%
        pull(n) %>%
        max()
      
      n_repli_binary_all_outcome_success_all <- repli_binary_orig %>%
        left_join(repli_outcomes_orig %>% select(report_id, score = repli_score_criteria_met), by = "report_id") %>%
        drop_na() %>%
        nrow()
    }
    
    # Replication outcomes by discipline
    {
      n_journals_crim <- publications %>%
        group_by(COS_pub_expanded) %>%
        dplyr::summarize(n = n()) %>%
        filter(COS_pub_expanded == "criminology") %>%
        pull(n)
      
      n_journals_econ_exclusive <- publications %>%
        filter(COS_pub_expanded == "economics") %>%
        mutate(publication_standard = str_to_lower(publication_standard)) %>%
        filter(!str_detect(publication_standard, "financ")) %>%
        nrow()
      
      n_journals_fin <- publications %>%
        mutate(publication_standard = str_to_lower(publication_standard)) %>%
        filter(str_detect(publication_standard, "financ")) %>%
        nrow()
      
      n_journals_health <- publications %>%
        group_by(COS_pub_expanded) %>%
        dplyr::summarize(n = n()) %>%
        filter(COS_pub_expanded == "health") %>%
        pull(n)
      
      n_journals_manag <- publications %>%
        group_by(COS_pub_expanded) %>%
        dplyr::summarize(n = n()) %>%
        filter(COS_pub_expanded == "management") %>%
        pull(n)
      
      n_journals_market <- publications %>%
        filter(COS_pub_expanded == "marketing/org behavior") %>%
        mutate(publication_standard = str_to_lower(publication_standard)) %>%
        filter(!str_detect(publication_standard, "organ")) %>%
        nrow()
      
      n_journals_org <- publications %>%
        filter(COS_pub_expanded == "marketing/org behavior") %>%
        mutate(publication_standard = str_to_lower(publication_standard)) %>%
        filter(str_detect(publication_standard, "organ")) %>%
        nrow()
      
      n_journals_psych_exclusive <- publications %>%
        group_by(COS_pub_expanded) %>%
        dplyr::summarize(n = n()) %>%
        filter(COS_pub_expanded == "psychology") %>%
        pull(n)
      
      n_journals_admin <- publications %>%
        group_by(COS_pub_expanded) %>%
        dplyr::summarize(n = n()) %>%
        filter(COS_pub_expanded == "public administration") %>%
        pull(n)
      
      n_journals_soc_exclusive <- publications %>%
        group_by(COS_pub_expanded) %>%
        dplyr::summarize(n = n()) %>%
        filter(COS_pub_expanded == "sociology") %>%
        pull(n)
      
    }
    
    # Multiple replications of single claims
    {
      n_multi_repli_same_claim <- 
        repli_outcomes_orig %>%
        filter(!is_covid & repli_type == "new data") %>%
        select(claim_id, paper_id, rr_id, repli_sample_size_value, contains("many")) %>%
        filter(is_manylabs == "posthoc" | is_manylabs == "traditional") %>%
        filter(manylabs_type != "aggregation") %>%
        group_by(claim_id, rr_id) %>%
        arrange(desc(repli_sample_size_value)) %>%
        dplyr::slice(1) %>%
        ungroup() %>%
        bind_rows(
          repli_outcomes %>%
            filter(!is_covid & repli_type == "secondary data") %>%
            select(claim_id, paper_id, rr_id, repli_sample_size_value, contains("many")) %>%
            filter(is_manylabs == "posthoc" | is_manylabs == "traditional") %>%
            filter(manylabs_type != "aggregation")
        ) %>%
        nrow()
    }
    
    # Hybrid replications
    {
      n_repli_hybrid <- repli_outcomes_orig %>%
        filter(!is_covid) %>%
        filter(repli_type == "original and secondary data") %>%
        nrow()
      
      n_repli_hybrid_pure <- repli_outcomes_orig %>%
        filter(!is_covid) %>%
        filter(repli_type == "original and secondary data") %>%
        semi_join(repli_outcomes_orig %>% filter(!is_covid & repli_type == "secondary data"), by = c("claim_id", "rr_id")) %>%
        nrow()
      
      n_repli_hybrid_impure <- repli_outcomes_orig %>%
        filter(!is_covid) %>%
        filter(repli_type == "original and secondary data") %>%
        anti_join(repli_outcomes_orig %>% filter(!is_covid & repli_type == "secondary data"), by = c("claim_id", "rr_id")) %>%
        nrow()
      
      n_repli_hybrid_impure_success <- repli_outcomes_orig %>%
        filter(!is_covid) %>%
        filter(repli_type == "original and secondary data") %>%
        anti_join(repli_outcomes_orig %>% filter(!is_covid & repli_type == "secondary data"), by = c("claim_id", "rr_id")) %>%
        filter(repli_score_criteria_met) %>%
        nrow()
      
      n_repli_hybrid_impure_fail <- repli_outcomes_orig %>%
        filter(!is_covid) %>%
        filter(repli_type == "original and secondary data") %>%
        anti_join(repli_outcomes_orig %>% filter(!is_covid & repli_type == "secondary data"), by = c("claim_id", "rr_id")) %>%
        filter(!repli_score_criteria_met) %>%
        nrow()
      
      p_repli_hybrid_impure_success <- 
        paste0(format.round(100*n_repli_hybrid_impure_success/n_repli_hybrid_impure,
                            0),"%")
      
      n_repli_hybrid_pure_success <- repli_outcomes_orig %>%
        filter(!is_covid) %>%
        filter(repli_type == "original and secondary data") %>%
        semi_join(repli_outcomes_orig %>% filter(!is_covid & repli_type == "secondary data"), by = c("claim_id", "rr_id")) %>%
        bind_rows(
          repli_outcomes_orig %>%
            filter(!is_covid & repli_type == "secondary data") %>%
            semi_join(repli_outcomes_orig %>%
                        filter(!is_covid) %>%
                        filter(repli_type == "original and secondary data"), by = c("claim_id", "rr_id"))
        ) %>%
        group_by(claim_id) %>%
        dplyr::summarize(t = sum(repli_score_criteria_met)) %>%
        count(t) %>%
        filter(t == 2) %>%
        pull(n)
      
      p_repli_hybrid_pure_success <- 
        paste0(format.round(100*n_repli_hybrid_pure_success/n_repli_hybrid_pure,
                            0),"%")
      
      n_repli_hybrid_pure_fail_both <- repli_outcomes_orig %>%
        filter(!is_covid) %>%
        filter(repli_type == "original and secondary data") %>%
        semi_join(repli_outcomes_orig %>% filter(!is_covid & repli_type == "secondary data"), by = c("claim_id", "rr_id")) %>%
        bind_rows(
          repli_outcomes_orig %>%
            filter(!is_covid & repli_type == "secondary data") %>%
            semi_join(repli_outcomes_orig %>%
                        filter(!is_covid) %>%
                        filter(repli_type == "original and secondary data"), by = c("claim_id", "rr_id"))
        ) %>%
        group_by(claim_id) %>%
        dplyr::summarize(t = sum(repli_score_criteria_met)) %>%
        count(t) %>%
        filter(t == 0) %>%
        pull(n)
      
      p_repli_hybrid_pure_fail_both <- 
        paste0(format.round(100*n_repli_hybrid_pure_fail_both/n_repli_hybrid_pure,
                            0),"%")
      
      n_repli_hybrid_pure_success_hybrid_fail_ind <- repli_outcomes_orig %>%
        filter(!is_covid) %>%
        filter(repli_type == "original and secondary data") %>%
        semi_join(repli_outcomes_orig %>% filter(!is_covid & repli_type == "secondary data"), by = c("claim_id", "rr_id")) %>%
        bind_rows(
          repli_outcomes_orig %>%
            filter(!is_covid & repli_type == "secondary data") %>%
            semi_join(repli_outcomes_orig %>%
                        filter(!is_covid) %>%
                        filter(repli_type == "original and secondary data"), by = c("claim_id", "rr_id"))
        ) %>%
        group_by(claim_id) %>%
        dplyr::summarize(t = sum(repli_score_criteria_met)) %>%
        count(t) %>%
        filter(t == 1) %>%
        pull(n)
      
      p_repli_hybrid_pure_success_hybrid_fail_ind <- 
        paste0(format.round(100*n_repli_hybrid_pure_success_hybrid_fail_ind/n_repli_hybrid_pure,
                            0),"%")
      
      n_repli_hybrid_replicated <- repli_outcomes_orig %>%
        filter(!is_covid) %>%
        filter(repli_type == "original and secondary data") %>%
        filter(repli_score_criteria_met) %>%
        nrow()
      
      p_repli_hybrid_replicated <- 
        paste0(format.round(100*n_repli_hybrid_replicated/n_repli_hybrid,
                            0),"%")
      
      n_repli_hybrid_ind_success <- 
        repli_outcomes_orig %>%
        filter(!is_covid) %>%
        filter(repli_type == "original and secondary data") %>%
        semi_join(repli_outcomes_orig %>% filter(!is_covid & repli_type == "secondary data"), by = c("claim_id", "rr_id")) %>%
        bind_rows(
          repli_outcomes_orig %>%
            filter(!is_covid & repli_type == "secondary data") %>%
            semi_join(repli_outcomes_orig %>%
                        filter(!is_covid) %>%
                        filter(repli_type == "original and secondary data"), by = c("claim_id", "rr_id"))
        ) %>%
        group_by(claim_id) %>%
        dplyr::summarize(t = sum(repli_score_criteria_met)) %>%
        count(t) %>%
        filter(t == 2) %>%
        pull(n)
      
      p_repli_hybrid_pure_ind_success <- 
        paste0(format.round(100*n_repli_hybrid_ind_success/n_repli_hybrid_pure,
                            0),"%")
        
    }
    
    # Statistical significance and effect size for replications completed from Phase 1 sample
    {
      repli_outcomes_p1 <- repli_outcomes %>% filter(type_internal!="p2") %>%
        group_by(paper_id) %>%
        mutate(weight = 1/n()) %>%
        ungroup()
      
      n_papers_p1 <- length(unique(repli_outcomes_p1$paper_id))
      
      n_claims_p1 <- length(unique(repli_outcomes_p1$claim_id))
      
      n_papers_repli_stat_sig_claims_same_dir_p1 <- 
        format.round(sum(as.numeric((!is.na(repli_outcomes_p1$repli_p_value) & repli_outcomes_p1$repli_p_value <= 0.05 & 
                                       !is.na(repli_outcomes_p1$repli_pattern_criteria_met) & repli_outcomes_p1$repli_pattern_criteria_met==TRUE))*
                           repli_outcomes_p1$weight
        ),1)
      
      p_papers_repli_stat_sig_claims_same_dir_p1 <- 
        bootstrap.clust(data=repli_outcomes_p1,
                        FUN=function(x) {
                          x <- x %>% group_by(paper_id) %>% mutate(weight = 1/n())
                          weighted.mean((!is.na(x$repli_p_value) & x$repli_p_value <= 0.05 &
                                           !is.na(x$repli_pattern_criteria_met) 
                                         & x$repli_pattern_criteria_met==TRUE),
                                        x$weight)
                        },
                        clustervar = "paper_id",
                        keepvars=c("repli_p_value","repli_pattern_criteria_met"),
                        alpha=.05,tails="two-tailed",iters=iters,
                        format.percent=TRUE,digits=1,na.rm = FALSE
        )$formatted.text
      
      
      n_papers_repli_stat_sig_claims_opp_dir_p1 <- 
        format.round(sum(as.numeric((!is.na(repli_outcomes_p1$repli_p_value) & repli_outcomes_p1$repli_p_value <= 0.05 & 
                                       !is.na(repli_outcomes_p1$repli_pattern_criteria_met) & repli_outcomes_p1$repli_pattern_criteria_met==FALSE))*
                           repli_outcomes_p1$weight
        ),1)
      
      p_papers_repli_stat_sig_claims_opp_dir_p1 <- 
        bootstrap.clust(data=repli_outcomes_p1,
                        FUN=function(x) {
                          x <- x %>% group_by(paper_id) %>% mutate(weight = 1/n())
                          weighted.mean((!is.na(x$repli_p_value) & x$repli_p_value <= 0.05 &
                                           !is.na(x$repli_pattern_criteria_met) & x$repli_pattern_criteria_met==FALSE),
                                        x$weight)
                        },
                        clustervar = "paper_id",
                        keepvars=c("repli_p_value","repli_pattern_criteria_met"),
                        alpha=.05,tails="two-tailed",iters=iters,
                        format.percent=TRUE,digits=1,na.rm = FALSE
        )$formatted.text
      
      n_papers_repli_non_stat_sig_p1 <- 
        format.round(sum(as.numeric(!is.na(repli_outcomes_p1$repli_p_value) & repli_outcomes_p1$repli_p_value > 0.05)*
                           repli_outcomes_p1$weight),1)
      
      p_papers_repli_non_stat_sig_p1 <- 
        bootstrap.clust(data=repli_outcomes_p1,
                        FUN=function(x) {
                          x <- x %>% group_by(paper_id) %>% mutate(weight = 1/n())
                          weighted.mean(!is.na(x$repli_p_value) & as.numeric(x$repli_p_value > 0.05),
                                        x$weight)
                        },
                        clustervar = "paper_id",
                        keepvars=c("repli_p_value","repli_pattern_criteria_met"),
                        alpha=.05,tails="two-tailed",iters=iters,
                        format.percent=TRUE,digits=1,na.rm = FALSE
        )$formatted.text
      
      n_claims_repli_stat_sig_claims_same_dir_p1 <- 
        sum(as.numeric((!is.na(repli_outcomes_p1$repli_p_value) & repli_outcomes_p1$repli_p_value <= 0.05 & 
                          !is.na(repli_outcomes_p1$repli_pattern_criteria_met) & repli_outcomes_p1$repli_pattern_criteria_met==TRUE))
        )
      
      p_claims_repli_stat_sig_claims_same_dir_p1 <- 
        format.text.percent(sum(as.numeric((!is.na(repli_outcomes_p1$repli_p_value) & repli_outcomes_p1$repli_p_value <= 0.05 & 
                                              !is.na(repli_outcomes_p1$repli_pattern_criteria_met) & repli_outcomes_p1$repli_pattern_criteria_met==TRUE))),
                            n_claims)
      
      n_claims_repli_stat_sig_claims_opp_dir_p1 <- 
        sum(as.numeric((!is.na(repli_outcomes_p1$repli_p_value) & repli_outcomes_p1$repli_p_value <= 0.05 & 
                          !is.na(repli_outcomes_p1$repli_pattern_criteria_met) & repli_outcomes_p1$repli_pattern_criteria_met==FALSE))
        )
      
      p_claims_repli_stat_sig_claims_opp_dir_p1 <- 
        format.text.percent(sum(as.numeric((!is.na(repli_outcomes_p1$repli_p_value) & repli_outcomes_p1$repli_p_value <= 0.05 & 
                                              !is.na(repli_outcomes_p1$repli_pattern_criteria_met) & repli_outcomes_p1$repli_pattern_criteria_met==FALSE))),
                            n_claims)
      
      n_claims_repli_non_stat_sig_p1 <- 
        sum(as.numeric(!is.na(repli_outcomes_p1$repli_p_value) & repli_outcomes_p1$repli_p_value > 0.05))
      
      p_claims_repli_non_stat_sig_p1 <- 
        format.text.percent(sum(as.numeric(!is.na(repli_outcomes_p1$repli_p_value) & repli_outcomes_p1$repli_p_value > 0.05)),
                            n_claims)
      
      table_s11_median_R_papers_neg_ratio <- 
        paste0(format.round(100*(table_s11_median_4_1-table_s11_median_4_2)/table_s11_median_4_1,1),"%")
      
      table_s11_median_R_claims_neg_ratio <- 
        paste0(format.round(100*(table_s11_median_4_3-table_s11_median_4_4)/table_s11_median_4_3,1),"%")
    }
    
    # Statistical significance and effect size for replications completed from Phase 2 sample
    {
      repli_outcomes_p2 <- repli_outcomes %>% filter(type_internal=="p2") %>%
        group_by(paper_id) %>%
        mutate(weight = 1/n()) %>%
        ungroup()
      
      n_papers_p2 <- length(unique(repli_outcomes_p2$paper_id))
      
      n_claims_p2 <- length(unique(repli_outcomes_p2$claim_id))
      
      n_papers_repli_stat_sig_claims_same_dir_p2 <- 
        format.round(sum(as.numeric((!is.na(repli_outcomes_p2$repli_p_value) & repli_outcomes_p2$repli_p_value <= 0.05 & 
                                       !is.na(repli_outcomes_p2$repli_pattern_criteria_met) & repli_outcomes_p2$repli_pattern_criteria_met==TRUE))*
                           repli_outcomes_p2$weight
        ),1)
      
      p_papers_repli_stat_sig_claims_same_dir_p2 <- 
        bootstrap.clust(data=repli_outcomes_p2,
                        FUN=function(x) {
                          x <- x %>% group_by(paper_id) %>% mutate(weight = 1/n())
                          weighted.mean((!is.na(x$repli_p_value) & x$repli_p_value <= 0.05 &
                                           !is.na(x$repli_pattern_criteria_met) & x$repli_pattern_criteria_met==TRUE),
                                        x$weight)
                        },
                        clustervar = "paper_id",
                        keepvars=c("repli_p_value","repli_pattern_criteria_met"),
                        alpha=.05,tails="two-tailed",iters=iters,
                        format.percent=TRUE,digits=1,na.rm = FALSE
        )$formatted.text
      
      
      n_papers_repli_stat_sig_claims_opp_dir_p2 <- 
        format.round(sum(as.numeric((!is.na(repli_outcomes_p2$repli_p_value) & repli_outcomes_p2$repli_p_value <= 0.05 & 
                                       !is.na(repli_outcomes_p2$repli_pattern_criteria_met) & repli_outcomes_p2$repli_pattern_criteria_met==FALSE))*
                           repli_outcomes_p2$weight
        ),1)
      
      p_papers_repli_stat_sig_claims_opp_dir_p2 <- 
        bootstrap.clust(data=repli_outcomes_p2,
                        FUN=function(x) {
                          x <- x %>% group_by(paper_id) %>% mutate(weight = 1/n())
                          weighted.mean((!is.na(x$repli_p_value) & x$repli_p_value <= 0.05 &
                                           !is.na(x$repli_pattern_criteria_met) & x$repli_pattern_criteria_met==FALSE),
                                        x$weight)
                        },
                        clustervar = "paper_id",
                        keepvars=c("repli_p_value","repli_pattern_criteria_met"),
                        alpha=.05,tails="two-tailed",iters=iters,
                        format.percent=TRUE,digits=1,na.rm = FALSE
        )$formatted.text
      
      n_papers_repli_non_stat_sig_p2 <- 
        format.round(sum(as.numeric(!is.na(repli_outcomes_p2$repli_p_value) & repli_outcomes_p2$repli_p_value > 0.05)*
                           repli_outcomes_p2$weight),1)
      
      p_papers_repli_non_stat_sig_p2 <- 
        bootstrap.clust(data=repli_outcomes_p2,
                        FUN=function(x) {
                          x <- x %>% group_by(paper_id) %>% mutate(weight = 1/n())
                          weighted.mean(!is.na(x$repli_p_value) & as.numeric(x$repli_p_value > 0.05),
                                        x$weight)
                        },
                        clustervar = "paper_id",
                        keepvars=c("repli_p_value","repli_pattern_criteria_met"),
                        alpha=.05,tails="two-tailed",iters=iters,
                        format.percent=TRUE,digits=1,na.rm = FALSE
        )$formatted.text
      
      n_claims_repli_stat_sig_claims_same_dir_p2 <- 
        sum(as.numeric((!is.na(repli_outcomes_p2$repli_p_value) & repli_outcomes_p2$repli_p_value <= 0.05 & 
                          !is.na(repli_outcomes_p2$repli_pattern_criteria_met) & repli_outcomes_p2$repli_pattern_criteria_met==TRUE))
        )
      
      p_claims_repli_stat_sig_claims_same_dir_p2 <- 
        format.text.percent(sum(as.numeric((!is.na(repli_outcomes_p2$repli_p_value) & repli_outcomes_p2$repli_p_value <= 0.05 & 
                                              !is.na(repli_outcomes_p2$repli_pattern_criteria_met) & repli_outcomes_p2$repli_pattern_criteria_met==TRUE))),
                            n_claims)
      
      n_claims_repli_stat_sig_claims_opp_dir_p2 <- 
        sum(as.numeric((!is.na(repli_outcomes_p2$repli_p_value) & repli_outcomes_p2$repli_p_value <= 0.05 & 
                          !is.na(repli_outcomes_p2$repli_pattern_criteria_met) & repli_outcomes_p2$repli_pattern_criteria_met==FALSE))
        )
      
      p_claims_repli_stat_sig_claims_opp_dir_p2 <- 
        format.text.percent(sum(as.numeric((!is.na(repli_outcomes_p2$repli_p_value) & repli_outcomes_p2$repli_p_value <= 0.05 & 
                                              !is.na(repli_outcomes_p2$repli_pattern_criteria_met) & repli_outcomes_p2$repli_pattern_criteria_met==FALSE))),
                            n_claims)
      
      n_claims_repli_non_stat_sig_p2 <- 
        sum(as.numeric(!is.na(repli_outcomes_p2$repli_p_value) & repli_outcomes_p2$repli_p_value > 0.05))
      
      p_claims_repli_non_stat_sig_p2 <- 
        format.text.percent(sum(as.numeric(!is.na(repli_outcomes_p2$repli_p_value) & repli_outcomes_p2$repli_p_value > 0.05)),
                            n_claims)
      
      table_s10_median_R_papers_neg_ratio <- 
        paste0(format.round(100*(table_s10_median_4_1-table_s10_median_4_2)/table_s10_median_4_1,1),"%")
      
      table_s10_median_R_claims_neg_ratio <- 
        paste0(format.round(100*(table_s10_median_4_3-table_s10_median_4_4)/table_s10_median_4_3,1),"%")
    }
    
    # Different protocol
    {
      n_claims_same_claim_different_protocol <- 
        repli_outcomes %>%
        filter(!is_covid) %>%
        filter(is_manylabs == "posthoc") %>%
        mutate(
          claim_id = ifelse(claim_id == "AgO1_gdvqwq", "AgO1_single-trace", claim_id),
          claim_id = ifelse(claim_id == "AvOr_gdyrdr", "AvOr_single-trace", claim_id),
          claim_id = ifelse(claim_id == "zlBL_56zxpp", "zlBL_single-trace", claim_id),
        ) %>%
        count(claim_id) %>%
        nrow()
      
      n_studies_same_claim_different_protocol <- 
        repli_outcomes_orig %>%
        filter(!is_covid) %>%
        filter(is_manylabs == "posthoc") %>%
        filter(manylabs_type != "aggregation") %>%
        mutate(
          claim_id = ifelse(claim_id == "AgO1_gdvqwq", "AgO1_single-trace", claim_id),
          claim_id = ifelse(claim_id == "AvOr_gdyrdr", "AvOr_single-trace", claim_id),
          claim_id = ifelse(claim_id == "zlBL_56zxpp", "zlBL_single-trace", claim_id),
        ) %>%
        count(rr_id) %>%
        nrow()
    }
    
    # Completion by journal (caption for Figure s8)
    {
      all_sourced <- rr_type_key %>% 
        semi_join(paper_status_tracking %>% filter(p1_delivery), by = "paper_id") %>% 
        filter(type == "replication" | type == "hybrid") %>% 
        select(paper_id) %>% 
        distinct()
      
      journal_completion <- paper_status_tracking %>% 
        filter(RR) %>% 
        select(paper_id) %>% 
        left_join(paper_metadata %>% select(paper_id, journal = publication_standard), by = "paper_id") %>% 
        mutate(papers_not_sourced = !(paper_id %in% all_sourced$paper_id)) %>% 
        mutate(finished = paper_id %in% repli_all_cases$paper_id) %>% 
        mutate(
          registered = !finished & paper_id %in% (rr_internal_tracking %>% filter(str_detect(type, "Replication")) %>% 
                                                    filter(!is.na(registrations) | prereg_completion == "approve") %>% pull(paper_id))
        ) %>% 
        mutate(
          completed_prereg = !finished & !registered & paper_id %in% (rr_internal_tracking %>% 
                                                                        filter(str_detect(type, "Replication")) %>% filter(prereg_completion == "complete") %>% pull(paper_id))
        ) %>% 
        mutate(
          partial_prereg = !finished & !registered & !completed_prereg & paper_id %in% (rr_internal_tracking %>% 
                                                                                          filter(str_detect(type, "Replication")) %>% filter(prereg_completion == "partial" | osf_activity) %>% pull(paper_id))
        ) %>% 
        mutate(not_started = !papers_not_sourced & !finished & !registered & !completed_prereg & !partial_prereg) %>%
        group_by(journal) %>%
        dplyr::summarise(n=n())
      
      n_papers_per_journal_min <- min(journal_completion$n)
      n_papers_per_journal_max <- max(journal_completion$n)
    }
  }
  
  
  # Clean up input values and export
  {
    rm(iters,orig_outcomes,repli_outcomes,repli_outcomes_merged)
    return(rev(as.list(environment())))
  }
}

# Generate figures
figures <- function(iters = 100,generate_binary_outcomes_data=FALSE){
  # Load libraries
  {
    library(dplyr)
    library(ggplot2)
    library(ggExtra)
    library(DT)
    library(tidyr)
    library(stringr)
    library(Hmisc)
    library(scales)
    library(wCorr)
    library(corrplot)
    library(cowplot)
    library(ggridges) #note: using the github version, as the CRAN hasn't been pushed to get the weights functionality
    library(ggside)
    library(weights)
    library(glue)
    library(forcats)
  }
  
  # Load data and common functions
  {
    # Set defaults for convenience
    {
      if (!exists("iters")){ iters <- 100}
      if (!exists("generate_binary_outcomes_data")){ generate_binary_outcomes_data <- FALSE}
    }
    
    # Check if loading locally
    if(file.exists("common functions.R") & file.exists("analyst data.RData")){
      load("analyst data.RData")
      source("common functions.R")
      source("repli_binary.R")
    } else {
      load("Analysis/Paper 5/Code and data/Analyst package/analyst data.RData")
      source("Analysis/Paper 5/Code and data/Analyst package/common functions.R")
      source("Analysis/Paper 5/Code and data/Analyst package/repli_binary.R")
    }
  }

  # Initialization and data preparation
  {
    # Trim out non-version of record lines and non-COVID entries
    {
      repli_outcomes_orig <- repli_outcomes
      
      repli_outcomes <- repli_outcomes[repli_outcomes$repli_version_of_record,]
      repli_outcomes <- repli_outcomes[!repli_outcomes$is_covid,]
    }
    
    # Generate binary data. Note: generate_binary_data flag by default is set 
    # to FALSE, which pulls the binary data precalculated from the data pipeline.
    # If set to TRUE, the code will run to generate the binary success measures
    # from scratch. This is time consuming, but will produce identical results)
    {
      if(generate_binary_outcomes_data==TRUE ){
        # Overwrite static version from scratch
        library(metafor)
        library(pwr)
        library(purrr)
        library(BFF)
        library(BayesFactor)
        library(effectsize)
        library(BayesRep)
        library(metaBMA)
        library(ReplicationSuccess)
        repli_binary <- calculate_repli_binary(repli_outcomes, 
                                               orig_outcomes, 
                                               paper_metadata)
      } else {
        repli_binary_orig <- repli_binary
        # Switch to paper_id / claim_id coding
        
      }
      
    }
    
    # Add variables to repli_binary
    {
      repli_binary <- merge(repli_outcomes[c("paper_id","claim_id","report_id","repli_score_criteria_met")],
                            repli_binary,all.x = FALSE,all.y=TRUE,by="report_id")
    }
    
    # Set binary measures variable names
    {
      fields.raw <-
        c("political science",
          "economics and finance",
          "sociology and criminology",
          "psychology and health",
          "business",
          "education")
      fields.format.2.row <- 
        c("Political\nScience",
          "Economics\nand Finance",
          "Sociology and\nCriminology",
          "Psychology\nand Health",
          "Business",
          "Education")
      fields.format.3.row <- 
        c("Political\nScience",
          "Economics\nand\nFinance",
          "Sociology\nand\nCriminology",
          "Psychology\nand\nHealth",
          "Business",
          "Education")
      fields.abbreviated <- 
        c("Political Science",
          "Economics",
          "Sociology",
          "Psychology",
          "Business",
          "Education")
      
      binvars.raw <- c("repli_score_criteria_met",
                       "repli_binary_analyst",
                       "repli_binary_orig_wthn",
                       "repli_binary_rep_wthn",
                       "repli_binary_wthn_pi",
                       "repli_binary_meta_success",
                       "repli_binary_sum_p",
                       "repli_binary_telescopes",
                       "repli_binary_bf_result",
                       "repli_binary_bayes_rep",
                       "repli_binary_bma_result",
                       "repli_binary_skep_p",
                       "repli_binary_correspondence")
      binvars.order <- binvars.raw
      
      binvars.full.text <- c("Statistical signifance + pattern",
                         "Subjective interpretation",
                         "Original effect in replication's CI",
                         "Replication effect in original's CI",
                         "Replication effect in prediction interval",
                         "Meta-analysis",
                         "Sum of p-values",
                         "Small telescopes",
                         "Bayes factor",
                         "Replication Bayes factor",
                         "Bayesian meta-analysis",
                         "Skeptical p-value",
                         "Correspondence test")
      binvars.group <- c("Statistic significance / p-value based measures",
                         "SCORE specific",
                         "Contained within interval measures",
                         "Contained within interval measures",
                         "Contained within interval measures",
                         "Meta analysis based measures",
                         "Statistic significance / p-value based measures",
                         "Combined effect magnitude and uncertainty measures",
                         "Combined effect magnitude and uncertainty measures",
                         "Combined effect magnitude and uncertainty measures",
                         "Meta analysis based measures",
                         "Statistic significance / p-value based measures",
                         "Combined effect magnitude and uncertainty measures")
      
      binvars.order <- c("repli_binary_analyst",
                         "repli_score_criteria_met",
                         "repli_binary_sum_p",
                         "repli_binary_skep_p",
                         "repli_binary_orig_wthn",
                         "repli_binary_rep_wthn",
                         "repli_binary_wthn_pi",
                         "repli_binary_meta_success",
                         "repli_binary_bma_result",
                         "repli_binary_telescopes",
                         "repli_binary_bf_result",
                         "repli_binary_bayes_rep",
                         "repli_binary_correspondence")
      
      df.binvars <- data.frame(binvars.raw,binvars.full.text,binvars.group)
      df.binvars <- do.call(rbind,lapply(binvars.order,function(x) {
        df.binvars[df.binvars$binvars.raw==x,]
      }))
    }
  }
  
  # Main text figures
  {
    # Figure 1: Replication success rates across 13 binary assessments for papers
    {
      # Data wrangling
      {
        # repli_binary <- repli_outcomes[c("paper_id", "claim_id",
        #                                  df.binvars$binvars.raw)]
        
        # Summarize by proportions w/ CIs
        binary.proportions <- 
          do.call(rbind,lapply(df.binvars$binvars.raw,function(binary.var){
            repli_binary_temp <- na.omit(repli_binary[c("paper_id",binary.var)])
            repli_binary_temp <- repli_binary_temp %>%
              group_by(paper_id) %>%
              mutate(weight = 1/n())
            
            n_testable_claims <- nrow(repli_binary_temp)
            p_testable_claims <- n_testable_claims/length(unique(repli_outcomes$claim_id))
            n_passed_claims <- sum(repli_binary_temp[[binary.var]])
            p_passed_temp <- binconf(n_passed_claims,n_testable_claims)
            p_passed_claims <- p_passed_temp[1]
            p_passed_CI_LB_claims <- p_passed_temp[2]
            p_passed_CI_UB_claims <- p_passed_temp[3]
            
            n_testable_papers <- length(unique((repli_binary_temp$paper_id)))
            p_testable_papers <- n_testable_papers/length(unique(repli_outcomes$paper_id))
            n_passed_papers <- sum(repli_binary_temp[[binary.var]]*repli_binary_temp$weight)
            p_passed_temp <- cw.proportion(as.integer(repli_binary_temp[[binary.var]]),
                                           repli_binary_temp$weight,repli_binary_temp$paper_id,iters)
            p_passed_papers <- p_passed_temp$point.estimate
            p_passed_CI_LB_papers <- p_passed_temp$CI.lb
            p_passed_CI_UB_papers <- p_passed_temp$CI.ub
            
            data.frame(binary.var,
                       n_testable_claims,p_testable_claims,n_passed_claims,p_passed_claims,p_passed_CI_LB_claims,p_passed_CI_UB_claims,
                       n_testable_papers,p_testable_papers,n_passed_papers,p_passed_papers,p_passed_CI_LB_papers,p_passed_CI_UB_papers
            )
          }))
        
        binary.proportions$binary.var <- ordered(binary.proportions$binary.var,
                                                 levels = binvars.raw,
                                                 labels= binvars.full.text)
        
          
        # Reorder by size
        binary.proportions <- binary.proportions[order(-binary.proportions$p_passed_papers),]
        binary.proportions$binary.var <- ordered(as.character(binary.proportions$binary.var),
                                                 levels = binary.proportions$binary.var,
                                                 labels= binary.proportions$binary.var)
        
        binary.proportions$num <- rev(1:nrow(binary.proportions))
        
        binary.proportions$n.text <- paste0("n=",format.round(binary.proportions$n_passed_papers,1),"/",binary.proportions$n_testable_papers)
        
        binary.proportions$num.claims <- rev(1:nrow(binary.proportions))
        
        binary.proportions$n.text.claims <- paste0("n=",format.round(binary.proportions$n_passed_claims,0),"/",binary.proportions$n_testable_claims)
      }
      
      # Chart generation
      {
        #ggplot(binary.proportions, aes(x=binary.var,y=PointEst)) + 
        plot <- ggplot(binary.proportions, aes(x=reorder(binary.var, -p_passed_papers),y=p_passed_papers)) + 
          geom_bar(stat="identity",fill="grey90") +
          geom_text(aes(x=num,y=0.02,label=n.text),hjust=0,size=3)+
          funkyheatmap::geom_rounded_rect(
            aes(xmin=num-.45,xmax=num+.45,ymin = p_passed_CI_LB_papers, ymax = p_passed_CI_UB_papers),
            radius=unit(3, "points"),show.legend=FALSE,color="black",size=0,fill=palette_score_charts[7])+
          geom_segment(aes(x=num-.45,xend=num+.45,y = p_passed_papers, yend = p_passed_papers),color="white")+
          # geom_errorbar(aes(y=PointEst,ymin=Lower, ymax=Upper), width=.2)+
          theme_minimal()+
          scale_x_discrete(limits=rev(binary.proportions$binary.var))+
          coord_flip()+
          scale_y_continuous(expand=expansion(add = c(0, .05)),
                             labels = scales::percent_format(),
                             breaks=c(0,.25,.5,.50,.75,1),
                             limits=c(0,1))+
          theme(legend.position = "none",
                legend.title=element_blank(),
                panel.border = element_blank(),
                panel.grid = element_blank(),
                axis.line = element_blank(),
                text = element_text(family = "Arial")
          )+
          xlab("Binary success measure")+
          ylab("Proportion successfully replicated according to measure")
        
        figure_1 <- bundle_ggplot(
          plot = plot,
          width = 3000,height = 1000,units = "px",bg="white")
      }
    }

    # Figure 2: Correlation matrix among binary assessments of replication success across papers
    {
      # Data wrangling
      {
        # repli_binary <- repli_outcomes[c("paper_id", "claim_id",
        #                                  df.binvars$binvars.raw)]

        # Summarize by proportions w/ CIs
        repli_binary_binaries <- repli_binary[,3:ncol(repli_binary)]
        repli_binary_binaries <- repli_binary_binaries[c(binvars.order)]
        colnames(repli_binary_binaries) <- df.binvars$binvars.full.text
        
        # Generate a results matrix
        repli_binary_cors <- matrix(NA,nrow=ncol(repli_binary_binaries),ncol=ncol(repli_binary_binaries))
        rownames(repli_binary_cors) <- colnames (repli_binary_cors) <- colnames(repli_binary_binaries)
        
        get_wt_corr <- function(i.x,i.y){
          df.cors <- repli_binary_binaries[,c(i.x,i.y)]
          df.cors$paper_id <- repli_binary$paper_id
          df.cors <- na.omit(df.cors)
          df.cors <- df.cors %>%
            group_by(paper_id) %>%
            mutate(weight = 1/n())
          
          weightedCorr(as.integer(unlist(df.cors[,1])),as.integer(unlist(df.cors[,2])),weights=df.cors$weight,method="Pearson")
        }
        
        # Get pairwise complete weighted correlations
        for(i in 1:nrow(repli_binary_cors)){
          for(j in 1:(i)){
            if(i==j){
              repli_binary_cors[i,j] <- 1
            } else{
              repli_binary_cors[j,i] <- get_wt_corr(i,j)
              repli_binary_cors[i,j] <- NA
            }
          }
        }
        
        binvars.x <- rownames(repli_binary_cors)
        binvars.y <- rownames(repli_binary_cors)
        df.chart <- expand.grid(binvars.x=binvars.x, binvars.y=binvars.y)
        df.chart$corr <-c(repli_binary_cors)
        df.chart$upper_text_label <- str_sub(ifelse(is.na(df.chart$corr) | df.chart$binvars.x==df.chart$binvars.y,"",
                                                    format.round(df.chart$corr,2)), 2, -1)
        df.chart$corrs.all <- ifelse(is.na(df.chart$corr) | df.chart$binvars.x==df.chart$binvars.y,NA,
                                     df.chart$corr)
        
      }
      
      # Chart generation
      {
        chart.palette <- palette_score_charts[7]
        
        top <- ggplot(df.chart, aes(x=binvars.x, y=binvars.y, fill= corr)) + 
          geom_tile()+
          geom_text(aes(x=binvars.y, y=binvars.x, label=upper_text_label),size=3,color="grey30")+
          scale_x_discrete(position = "top") +
          scale_y_discrete(limits=rev)+
          theme_minimal()+
          theme(
            plot.title = element_blank(),
            axis.title = element_blank(),
            axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=0,color="black"),
            axis.text.y = element_text(color="black"),
            legend.key.height= unit(1, 'cm'),
            panel.grid = element_blank(),
            legend.position="none"
          ) +
          scale_fill_gradient(low="white", high=chart.palette[1], limits=c(0, 1),na.value="white")
        
        df.chart$group <- "Distribution of\n all 144 correlations"
        bottom <- ggplot(df.chart, aes(x=corrs.all,y=group,fill=..x..)) + 
          #geom_density(fill=palette_score_charts[6])+
          geom_density_ridges_gradient(linetype=0)+
          scale_x_continuous(limits=c(0,1),breaks=seq(0,1,.1),expand=c(0,0))+
          scale_y_discrete(expand=c(0,0))+
          scale_fill_gradient( limits=c(0, 1),low="white", high=chart.palette[1])+
          geom_vline(xintercept=c(0,1))+
          #ylab("Distribution of\n all 144 correlations")+
          # scale_y_discrete(limits=rev)+
          theme_minimal()+
          theme(
            plot.title = element_blank(),
            axis.title = element_blank(),
            axis.text.y = element_text(angle = 0, vjust = 0, hjust=1 ,size=8),
            axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=0,size=6),
            legend.key.height= unit(1, 'cm'),
            panel.grid = element_blank(),
            legend.position="none",
            text = element_text(family = "Arial")
          )

        figure_2 <- bundle_ggplot(
          plot = plot_grid(top,bottom,ncol=1,rel_heights = c(5,1),align = "v"),
          width = 2000,height = 2000,units = "px",bg="white")
      }
      
    }
    
    # Figure 3: Scatterplot of Pearson’s R effect sizes for original (x-axis) and replication (y-axis) findings by papers.
    {
      # Data wrangling
      {
        all_effects <- repli_outcomes %>% 
          filter(!is_covid) %>% 
          filter(repli_version_of_record) %>% 
          select(paper_id,report_id, claim_id, repli_pattern_criteria_met, success = repli_score_criteria_met,
                 repli_conv_r, repli_conv_r_lb, repli_conv_r_ub) %>% 
          left_join(
            orig_outcomes %>% 
              select(claim_id, orig_conv_r, orig_conv_r_lb, orig_conv_r_ub) %>% 
              distinct(),
            by = "claim_id"
          ) %>% 
          drop_na(repli_conv_r) %>% 
          drop_na(orig_conv_r) %>% 
          mutate(across(contains("orig"), abs)) %>% 
          mutate(
            across(
              contains("repli"),
              function(x) ifelse(repli_pattern_criteria_met, abs(x), -1*abs(x))
            )
          ) %>%
          group_by(paper_id) %>%
          mutate(weight = 1/n()) %>%
          ungroup()
      }
      
      # Chart generation
      {
        chart.palette <- c(palette_score_charts[2], palette_score_charts[1])
        all_effects$weight.alpha <- all_effects$weight*1
        all_effects$weight.size <- all_effects$weight*1
      
        plot <- 
          ggplot(data = all_effects, aes(x = orig_conv_r, y = repli_conv_r, color = success,weight=weight)) +
          geom_point(aes(size = weight),alpha=.6, stroke=NA) +
          scale_size_continuous(range=c(.5,3),guide="none")+
          geom_rug() +
          scale_color_manual(labels = c("Failed", "Successful"),
                             values = chart.palette) +
          annotate("segment",x=0,y=0,xend=1,yend=1, linetype = 2, color = "slategray") +
          annotate("segment",x=0,y=0,xend=1,yend=0, linetype = "solid", color = "black") +
          annotate("segment",x=0,y=-.55,xend=0,yend=1, linetype = "solid", color = "black") +
          xlim(0, 1) +
          ylim(-0.55, 1) +
          labs(
            x = "Original",
            y = "Replication",
            color = "",
            title = ""
          ) +
          theme_minimal()+
          theme(
            plot.title = element_blank(),
            axis.title.x = element_text(size = 16),
            axis.text.x = element_text(size = 14),
            axis.title.y = element_text(size = 16),
            axis.text.y = element_text(size = 14),
            legend.text = element_text(size = 16),
            panel.grid = element_blank(),
            legend.position = "bottom"
          )+
          guides(color = guide_legend(override.aes = list(linetype = 0,size=6)))+
          geom_xsidedensity(alpha = .6,aes(fill=success),show.legend=FALSE,color="black")+
          geom_ysidedensity(alpha = .6,aes(fill=success),show.legend=FALSE,color="black")+
          theme(
            ggside.panel.border = element_blank(),
            ggside.panel.grid = element_blank(),
            ggside.panel.background = element_blank(),
            ggside.axis.text = element_blank(),
            ggside.axis.ticks = element_blank(),
            ggside.panel.scale = .3,
            text = element_text(family = "Arial")
          )+
          scale_fill_manual(labels = c("Failed", "Successful"),
                             values = chart.palette)
        
        figure_3 <- bundle_ggplot(
          plot = plot,
          width = 2000,height = 2000,units = "px",bg="white")
      }
    }
    
    # Figure 4: Scatterplot of Pearson’s r effect sizes for original and replication outcomes for new data (left) and secondary data (right) replication attempts
    {
      # Data wrangling
      {
        all_effects_nd <- repli_outcomes %>% 
          filter(!is_covid) %>% 
          filter(repli_version_of_record) %>% 
          filter(repli_type=="new data") %>% 
          select(paper_id,report_id, claim_id, repli_pattern_criteria_met, success = repli_score_criteria_met,
                 repli_conv_r, repli_conv_r_lb, repli_conv_r_ub) %>% 
          left_join(
            orig_outcomes %>% 
              select(claim_id, orig_conv_r, orig_conv_r_lb, orig_conv_r_ub) %>% 
              distinct(),
            by = "claim_id"
          ) %>% 
          drop_na(repli_conv_r) %>% 
          drop_na(orig_conv_r) %>% 
          mutate(across(contains("orig"), abs)) %>% 
          mutate(
            across(
              contains("repli"),
              function(x) ifelse(repli_pattern_criteria_met, abs(x), -1*abs(x))
            )
          ) %>%
          group_by(paper_id) %>%
          mutate(weight = 1/n()) %>%
          ungroup()
        
        all_effects_sd <- repli_outcomes %>% 
          filter(!is_covid) %>% 
          filter(repli_version_of_record) %>% 
          filter(repli_type=="secondary data") %>% 
          select(paper_id,report_id, claim_id, repli_pattern_criteria_met, success = repli_score_criteria_met,
                 repli_conv_r, repli_conv_r_lb, repli_conv_r_ub) %>% 
          left_join(
            orig_outcomes %>% 
              select(claim_id, orig_conv_r, orig_conv_r_lb, orig_conv_r_ub) %>% 
              distinct(),
            by = "claim_id"
          ) %>% 
          drop_na(repli_conv_r) %>% 
          drop_na(orig_conv_r) %>% 
          mutate(across(contains("orig"), abs)) %>% 
          mutate(
            across(
              contains("repli"),
              function(x) ifelse(repli_pattern_criteria_met, abs(x), -1*abs(x))
            )
          ) %>%
          group_by(paper_id) %>%
          mutate(weight = 1/n()) %>%
          ungroup()
      }
      
      # Chart generation
      {
        #chart.palette <- c("tomato3", "deepskyblue4")
        chart.palette <- c(palette_score_charts[2], palette_score_charts[1])
        all_effects_sd$weight.alpha <- all_effects_sd$weight*.6
        all_effects_nd$weight.alpha <- all_effects_nd$weight*.6
        
        panel_left <- ggplot(data = all_effects_nd, aes(x = orig_conv_r, y = repli_conv_r, color = success,weight=weight)) +
          geom_point(aes(size = weight),alpha=.6, stroke=NA) +
          scale_size_continuous(range=c(.5,3),guide="none")+
          geom_rug() +
          scale_color_manual(labels = c("Failed", "Successful"),
                             values = chart.palette) +
          annotate("segment",x=0,y=0,xend=1,yend=1, linetype = 2, color = "slategray") +
          annotate("segment",x=0,y=0,xend=1,yend=0, linetype = "solid", color = "black") +
          annotate("segment",x=0,y=-.55,xend=0,yend=1, linetype = "solid", color = "black") +
          xlim(0, 1) +
          ylim(-0.55, 1) +
          scale_x_continuous(breaks=seq(0,1,.25),labels=c("0",".25",".5",".50",1))+
          labs(
            x = "Original",
            y = "Replication",
            color = "",
            title = "New data"
          ) +
          theme_minimal()+
          theme(
            plot.title = element_text(size = 16),
            axis.title.x = element_text(size = 16),
            axis.text.x = element_text(size = 14),
            axis.title.y = element_text(size = 16),
            axis.text.y = element_text(size = 14),
            legend.text = element_text(size = 16),
            panel.grid = element_blank()
          )+
          guides(color = guide_legend(override.aes = list(linetype = 0)))+
          geom_xsidedensity(alpha = .6,aes(fill=success),show.legend=FALSE,color="black")+
          geom_ysidedensity(alpha = .6,aes(fill=success),show.legend=FALSE,color="black")+
          theme(
            ggside.panel.border = element_blank(),
            ggside.panel.grid = element_blank(),
            ggside.panel.background = element_blank(),
            ggside.axis.text = element_blank(),
            ggside.axis.ticks = element_blank(),
            ggside.panel.scale = .3,
            legend.position = "none",
            text = element_text(family = "Arial")
          )+
          scale_fill_manual(labels = c("Failed", "Successful"),
                            values = chart.palette)
        
        panel_right <- ggplot(data = all_effects_sd, aes(x = orig_conv_r, y = repli_conv_r, color = success,weight=weight)) +
          geom_point(aes(size = weight),alpha=.6, stroke=NA) +
          scale_size_continuous(range=c(.5,3),guide="none")+
          geom_rug() +
          scale_color_manual(labels = c("Failed", "Successful"),
                             values = chart.palette) +
          annotate("segment",x=0,y=0,xend=1,yend=1, linetype = 2, color = "slategray") +
          annotate("segment",x=0,y=0,xend=1,yend=0, linetype = "solid", color = "black") +
          annotate("segment",x=0,y=-.55,xend=0,yend=1, linetype = "solid", color = "black") +
          xlim(0, 1) +
          ylim(-0.55, 1) +
          scale_x_continuous(breaks=seq(0,1,.25),labels=c("0",".25",".5",".50",1))+
          labs(
            x = "Original",
            y = "Replication",
            color = "",
            title = "Secondary data"
          ) +
          theme_minimal()+
          theme(
            plot.title = element_text(size = 16),
            axis.title.x = element_text(size = 16),
            axis.text.x = element_text(size = 14),
            axis.title.y = element_text(size = 16),
            axis.text.y = element_text(size = 14),
            legend.text = element_text(size = 16),
            panel.grid = element_blank()
          )+
          guides(color = guide_legend(override.aes = list(linetype = 0)))+
          geom_xsidedensity(alpha = .6,aes(fill=success),show.legend=FALSE,color="black")+
          geom_ysidedensity(alpha = .6,aes(fill=success),show.legend=FALSE,color="black")+
          theme(
            ggside.panel.border = element_blank(),
            ggside.panel.grid = element_blank(),
            ggside.panel.background = element_blank(),
            ggside.axis.text = element_blank(),
            ggside.axis.ticks = element_blank(),
            ggside.panel.scale = .3,
            legend.position = "none",
            text = element_text(family = "Arial")
          )+
          scale_fill_manual(labels = c("Failed", "Successful"),
                            values = chart.palette)
        
        plot <- 
          plot_grid(plot_grid(panel_left,panel_right),
                  get_legend(ggplot(data = all_effects_sd, aes(x = orig_conv_r, y = repli_conv_r, color = success,weight=weight)) +
                               geom_point(aes(alpha = weight), size = 3) +
                               scale_alpha_continuous(range=c(.1,.6),guide="none")+
                               scale_color_manual(labels = c("Failed", "Successful"),
                                                  values = chart.palette)+
                               theme_minimal()+
                               theme(
                                 legend.title=element_blank(),
                                 panel.grid = element_blank(),
                                 legend.direction = "horizontal",
                                 legend.text = element_text(size = 16),
                                 text = element_text(family = "Arial")
                               )
                             ),nrow=2,rel_heights = c(10,1))
        
        figure_4 <- bundle_ggplot(
          plot = plot,
          width = 2000,height = 1200,units = "px",bg="white")

      }
    }

  }
  
  # Supplement figures
  {
    # Figure S2
    {
      plot <- 
        repli_outcomes_orig %>% 
        filter(!is_covid & repli_type != "original and secondary data") %>% # exclude covid and hybrids 
        filter(is_manylabs == "not manylabs" | manylabs_type != "aggregation") %>% 
        group_by(claim_id, rr_id) %>% 
        arrange(desc(repli_sample_size_value)) %>% # exclude stage 1 when stage 2 is available
        dplyr::slice(1) %>% 
        ungroup() %>% 
        mutate(p2 = nchar(rr_id) > 4) %>%
        mutate(type = select(., c(p2, repli_type)) %>% apply(1, function(x) str_c(x, collapse = "_"))) %>% 
        #select(report_id, type, power = repli_power_for_50_effect) %>% 
        select(report_id, type, power = combined_power_50) %>% 
        mutate(power = round(power*100, 0)) %>% 
        drop_na() %>% 
        ggplot(aes(x = power, fill = type)) +
        geom_density(alpha = 0.5) +
        scale_fill_manual(values = c("tomato", "deepskyblue", "tomato4", "deepskyblue4"), 
                          labels = c("P1 new data", "P1 secondary data", "P2 new data", "P2 secondary data")) +
        labs(
          x = "Power (%)",
          y = "",
          fill = ""
        ) +
        theme_light() +
        theme(
          axis.text.y = element_blank(),
          legend.position = "bottom",
          text = element_text(family = "Arial")
        )
      
      figure_s2 <- bundle_ggplot(
        plot = plot,
        width = 2000,height = 2000,units = "px",bg="white")
    }
    
    # Figure S3
    {
      plot <- repli_outcomes %>%
        filter(!is_covid & repli_version_of_record) %>%
        select(report_id, repli_type, repli_power_for_50_effect:rr_power_100_original_effect_alpha_.025_design) %>%
        select(-contains(".025")) %>%
        rename(main_50 = repli_power_for_50_effect, main_75 = repli_power_for_75_effect, main_100 = rr_power_100_original_effect) %>%
        rename(design_50 = rr_power_50_original_effect_design, design_75 = rr_power_75_original_effect_design, design_100 = rr_power_100_original_effect_design) %>%
        pivot_longer(cols = -c(report_id, repli_type), names_to = "measure", values_to = "value") %>%
        drop_na() %>%
        separate(measure, into = c("type", "threshold"), sep = "_") %>%
        pivot_wider(names_from = type, values_from = value) %>%
        drop_na() %>%
        filter(repli_type == "secondary data") %>%
        ggplot(aes(x = main, y = design, color = threshold)) +
        geom_point() +
        geom_smooth(method = "lm", se = F) +
        scale_color_manual(values = c("tomato3", "darkseagreen3", "deepskyblue3")) +
        xlim(0, 1) +
        ylim(0, 1) +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "slategray") +
        labs(x = "Traditional power", y = "SER power", color = "% of original effect", title = "Secondary data replications") +
        theme_light() +
        theme(
          panel.grid.major.x = element_blank(),
          panel.grid.major.y = element_line(color = "gray90"),
          axis.title = element_text(size = 22),
          axis.text = element_text(size = 20),
          plot.title = element_text(size = 26, hjust = 0.5),
          legend.text = element_text(size = 22),
          legend.position = "top",
          text = element_text(family = "Arial")
        )
      
      figure_s3 <- bundle_ggplot(
        plot = plot,
        width = 4000,height = 2000,units = "px",bg="white")
    }
    
    # Figure S4
    {
      plot <- repli_outcomes %>%
        filter(!is_covid & repli_version_of_record) %>%
        select(report_id, repli_type, repli_power_for_50_effect:rr_power_100_original_effect_alpha_.025_design) %>%
        select(-contains(".025")) %>%
        rename(main_50 = repli_power_for_50_effect, main_75 = repli_power_for_75_effect, main_100 = rr_power_100_original_effect) %>%
        rename(design_50 = rr_power_50_original_effect_design, design_75 = rr_power_75_original_effect_design, design_100 = rr_power_100_original_effect_design) %>%
        pivot_longer(cols = -c(report_id, repli_type), names_to = "measure", values_to = "value") %>%
        drop_na() %>%
        separate(measure, into = c("type", "threshold"), sep = "_") %>%
        pivot_wider(names_from = type, values_from = value) %>%
        drop_na() %>%
        filter(repli_type == "new data") %>%
        ggplot(aes(x = main, y = design, color = threshold)) +
        geom_point() +
        geom_smooth(method = "lm", se = F) +
        scale_color_manual(values = c("tomato3", "darkseagreen3", "deepskyblue3")) +
        xlim(0, 1) +
        ylim(0, 1) +
        geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "slategray") +
        labs(x = "Traditional power", y = "SER power", color = "% of original effect", title = "New data replications") +
        theme_light() +
        theme(
          panel.grid.major.x = element_blank(),
          panel.grid.major.y = element_line(color = "gray90"),
          axis.title = element_text(size = 22),
          axis.text = element_text(size = 20),
          plot.title = element_text(size = 26, hjust = 0.5),
          legend.text = element_text(size = 22),
          legend.position = "top",
          text = element_text(family = "Arial")
        )
      
      figure_s4 <- bundle_ggplot(
        plot = plot,
        width = 4000,height = 2000,units = "px",bg="white")
    }
    
    # Figure S5
    {
      plot <- repli_outcomes %>%
        filter(!is_covid & repli_version_of_record) %>%
        mutate(
          power_50 = ifelse(is.na(repli_power_for_50_effect), rr_power_50_original_effect_design, repli_power_for_50_effect),
          power_75 = ifelse(is.na(repli_power_for_75_effect), rr_power_75_original_effect_design, repli_power_for_75_effect),
          power_100 = ifelse(is.na(rr_power_100_original_effect), rr_power_100_original_effect_design, rr_power_100_original_effect)
        ) %>%
        select(
          report_id,
          repli_type,
          Traditional_50 = repli_power_for_50_effect,
          Traditional_75 = repli_power_for_75_effect,
          Traditional_100 = rr_power_100_original_effect,
          SER_50 = rr_power_50_original_effect_design,
          SER_75 = rr_power_75_original_effect_design,
          SER_100 = rr_power_100_original_effect_design,
          Combined_50 = power_50,
          Combined_75 = power_75,
          Combined_100 = power_100
        ) %>%
        pivot_longer(cols = -c(report_id, repli_type), names_to = "variable", values_to = "value") %>%
        separate(variable, into = c("type", "level"), sep = "_") %>%
        mutate(level = as_factor(level) %>% fct_relevel(., "50", "75", "100")) %>%
        mutate(
          level = case_match(
            level,
            "50" ~ "50% of original effect",
            "75" ~ "75% of original effect",
            "100" ~ "100% of original effect",
          )
        ) %>%
        ggplot(aes(x = value, fill = type)) +
        geom_density(alpha = 0.6) +
        scale_fill_manual(values = c("tomato3", "darkseagreen3", "deepskyblue3")) +
        xlim(0, 1) +
        labs(x = "Power", y = "", fill = "Variable version") +
        facet_wrap(~level) +
        theme_light() +
        theme(
          panel.grid.major.x = element_blank(),
          panel.grid.major.y = element_line(color = "gray90"),
          axis.title = element_text(size = 22),
          axis.text.x = element_text(size = 18),
          axis.text.y = element_blank(),
          plot.title = element_text(size = 26, hjust = 0.5),
          strip.text = element_text(color = "black", size = 20),
          strip.background = element_blank(),
          legend.text = element_text(size = 22),
          legend.position = "top",
          text = element_text(family = "Arial")
        )
      figure_s5 <- bundle_ggplot(
        plot = plot,
        width = 4000,height = 2000,units = "px",bg="white")
      
    }
    
    # Figure S6
    {
      plot <- rr_project_dates %>% 
        mutate(
          completed = case_when(
            completed > as.Date("2023-03-31") ~ as.Date("2023-03-31"),
            TRUE ~ completed
          )
        ) %>% 
        semi_join(repli_outcomes %>% filter(!is_covid & (manylabs_type != "aggregation" | is.na(manylabs_type))), by = "rr_id") %>% 
        mutate(duration = (completed - started) %>% as.numeric()) %>% 
        ggplot(aes(x = duration, fill = type)) +
        geom_density(alpha = 0.7) +
        scale_fill_manual(
          values = c("tomato1", "tomato4", "deepskyblue1", "deepskyblue4"),
          labels = c("New data, P1", "New data, P2", 
                     "Secondary data, P1", "Secondary data, P2")
        ) +
        labs(
          x = "Days",
          y = "",
          fill = "",
          title = "Duration of replication projects"
        ) +
        theme_light() +
        theme(
          plot.title = element_text(size = 11, hjust = 0.5),
          axis.text.y = element_blank(),
          legend.text = element_text(size = 11),
          legend.position = "bottom",
          text = element_text(family = "Arial")
        )
      
      figure_s6 <- bundle_ggplot(
        plot = plot,
        width = 2000,height = 2000,units = "px",bg="white")
      
    }
    
    # Figure S7
    {
      # Data wrangling
      {
        all_sourced <- rr_type_key %>% 
          semi_join(paper_status_tracking %>% filter(p1_delivery), by = "paper_id") %>% 
          filter(type == "replication" | type == "hybrid") %>% 
          select(paper_id) %>% 
          distinct()

        # Need all eligible papers to start
        data <- paper_status_tracking %>% 
          filter(RR) %>% 
          select(paper_id) %>% 
          left_join(paper_metadata %>% select(paper_id, field = COS_pub_category), by = "paper_id") %>% 
          mutate(papers_not_sourced = !(paper_id %in% all_sourced$paper_id)) %>% 
          mutate(finished = paper_id %in% repli_all_cases$paper_id) %>% 
          mutate(
            registered = !finished & paper_id %in% (rr_internal_tracking %>% filter(str_detect(type, "Replication")) %>% 
                                                      filter(!is.na(registrations) | prereg_completion == "approve") %>% pull(paper_id))
          ) %>% 
          mutate(
            completed_prereg = !finished & !registered & paper_id %in% (rr_internal_tracking %>% 
                                                                          filter(str_detect(type, "Replication")) %>% filter(prereg_completion == "complete") %>% pull(paper_id))
          ) %>% 
          mutate(
            partial_prereg = !finished & !registered & !completed_prereg & paper_id %in% (rr_internal_tracking %>% 
                                                                                            filter(str_detect(type, "Replication")) %>% filter(prereg_completion == "partial" | osf_activity) %>% pull(paper_id))
          ) %>% 
          mutate(not_started = !papers_not_sourced & !finished & !registered & !completed_prereg & !partial_prereg) %>% 
          pivot_longer(cols = -c(paper_id, field), names_to = "stage", values_to = "value") %>%
          arrange(field)
        
        n_papers_by_field <- data %>%
          mutate(field = str_to_sentence(field),
            field = case_match(
              field,
              "Economics and finance" ~ "Economics",
              "Psychology and health" ~ "Psychology",
              "Sociology and criminology" ~ "Sociology",
              .default = field
            )
          ) %>%
          group_by(field,paper_id) %>% 
          dplyr::summarize(n = 1) %>% 
          group_by(field) %>% 
          dplyr::summarize(n = sum(n)) %>%
          arrange(field)
          
        
        grouped_data <- data %>%
          group_by(field, stage) %>% 
          dplyr::summarize(t = sum(value) / n()) %>% 
          ungroup() %>% 
          mutate(field = str_to_sentence(field)) %>% 
          mutate(
            stage = case_match(
              stage,
              "papers_not_sourced" ~ "Never sourced",
              "not_started" ~ "Not started",
              "partial_prereg" ~ "Partial prereg or OSF",
              "completed_prereg" ~ "Completed prereg",
              "registered" ~ "Registered",
              "finished" ~ "Finished"
            )
          ) %>% 
          mutate(
            field = case_match(
              field,
              "Economics and finance" ~ "Economics",
              "Psychology and health" ~ "Psychology",
              "Sociology and criminology" ~ "Sociology",
              .default = field
            )
          ) %>% 
          mutate(stage = as_factor(stage) %>% fct_relevel(., "Never sourced", "Not started", "Partial prereg or OSF", 
                                                          "Completed prereg", "Registered", "Finished") %>% fct_rev())
        
        grouped_data$field <- factor(grouped_data$field,
                                     levels=sort(unique(grouped_data$field)),
                                     labels=paste0(sort(unique(grouped_data$field)),"\n(n=",n_papers_by_field$n,")")
                                     )
      }
      
      # Plot
      {
        plot <- ggplot(data=grouped_data,aes(x = field, y = t, fill = stage)) +
          scale_x_discrete(expand=c(0,0))+
          scale_y_continuous(expand=c(0,0))+
          geom_col(alpha = 0.7) +
          scale_fill_manual(values = c("mediumorchid4", "black", "tomato4", "seagreen4", "wheat4", "deepskyblue4")) +
          labs(
            x = "",
            y = "",
            fill = "",
            title = "Evidence set (n = 600)"
          ) +
          theme_light() +
          theme(
            plot.title = element_text(size = 11, hjust = 0.5),
            axis.text = element_text(size = 11),
            legend.text = element_text(size = 11),
            legend.position = "bottom",
            text = element_text(family = "Arial")
          )
      }
      
      figure_s7 <- bundle_ggplot(
        plot = plot,
        width = 2000,height = 2000,units = "px",bg="white")
    }
    
    # Figure S8
    {
      # Data wrangling
      {
        all_sourced <- rr_type_key %>% 
          semi_join(paper_status_tracking %>% filter(p1_delivery), by = "paper_id") %>% 
          filter(type == "replication" | type == "hybrid") %>% 
          select(paper_id) %>% 
          distinct()
        
        # need all eligible papers to start
        data <- paper_status_tracking %>% 
          filter(RR) %>% 
          select(paper_id) %>% 
          left_join(paper_metadata %>% select(paper_id, year = pub_year), by = "paper_id") %>% 
          mutate(papers_not_sourced = !(paper_id %in% all_sourced$paper_id)) %>% 
          mutate(finished = paper_id %in% repli_all_cases$paper_id) %>% 
          mutate(
            registered = !finished & paper_id %in% (rr_internal_tracking %>% filter(str_detect(type, "Replication")) %>% 
                                                      filter(!is.na(registrations) | prereg_completion == "approve") %>% pull(paper_id))
          ) %>% 
          mutate(
            completed_prereg = !finished & !registered & paper_id %in% (rr_internal_tracking %>% 
                                                                          filter(str_detect(type, "Replication")) %>% filter(prereg_completion == "complete") %>% pull(paper_id))
          ) %>% 
          mutate(
            partial_prereg = !finished & !registered & !completed_prereg & paper_id %in% (rr_internal_tracking %>% 
                                                                                            filter(str_detect(type, "Replication")) %>% filter(prereg_completion == "partial" | osf_activity) %>% pull(paper_id))
          ) %>% 
          mutate(not_started = !papers_not_sourced & !finished & !registered & !completed_prereg & !partial_prereg) %>% 
          pivot_longer(cols = -c(paper_id, year), names_to = "stage", values_to = "value")
        
        n_papers_by_year <- data %>%
          group_by(year,paper_id) %>% 
          dplyr::summarize(n = 1) %>% 
          group_by(year) %>% 
          dplyr::summarize(n = sum(n))
        
        grouped_data <- data %>%
          group_by(year, stage) %>% 
          dplyr::summarize(t = sum(value) / n()) %>% 
          ungroup() %>% 
          mutate(year = as.character(year)) %>% 
          mutate(
            stage = case_match(
              stage,
              "papers_not_sourced" ~ "Never sourced",
              "not_started" ~ "Not started",
              "partial_prereg" ~ "Partial prereg or OSF",
              "completed_prereg" ~ "Completed prereg",
              "registered" ~ "Registered",
              "finished" ~ "Finished"
            )
          ) %>% 
          mutate(stage = as_factor(stage) %>% fct_relevel(., "Never sourced", "Not started", "Partial prereg or OSF", 
                                                          "Completed prereg", "Registered", "Finished") %>% fct_rev())
        
        grouped_data$year <- factor(grouped_data$year,
                                     levels=sort(unique(grouped_data$year)),
                                     labels=paste0(sort(unique(grouped_data$year)),"\n(n=",n_papers_by_year$n,")")
        )
      }
      
      # Plot
      {
        plot <- ggplot(data=grouped_data,aes(x = year, y = t, fill = stage)) +
          scale_x_discrete(expand=c(0,0))+
          scale_y_continuous(expand=c(0,0))+
          geom_col(alpha = 0.7) +
          scale_fill_manual(values = c("mediumorchid4", "black", "tomato4", "seagreen4", "wheat4", "deepskyblue4")) +
          labs(
            x = "",
            y = "",
            fill = "",
            title = "Evidence set (n = 600)"
          ) +
          theme_light() +
          theme(
            plot.title = element_text(size = 11, hjust = 0.5),
            axis.text = element_text(size = 11),
            legend.text = element_text(size = 11),
            legend.position = "bottom",
            text = element_text(family = "Arial")
          )
        }
      
      figure_s8 <- bundle_ggplot(
        plot = plot,
        width = 2000,height = 2000,units = "px",bg="white")
    }
    
    # Figure S9
    {
      all_sourced <- rr_type_key %>% 
        semi_join(paper_status_tracking %>% filter(p1_delivery), by = "paper_id") %>% 
        filter(type == "replication" | type == "hybrid") %>% 
        select(paper_id) %>% 
        distinct()
      
      plot <- paper_status_tracking %>% 
        filter(RR) %>% 
        select(paper_id) %>% 
        left_join(paper_metadata %>% select(paper_id, journal = publication_standard), by = "paper_id") %>% 
        mutate(papers_not_sourced = !(paper_id %in% all_sourced$paper_id)) %>% 
        mutate(finished = paper_id %in% repli_all_cases$paper_id) %>% 
        mutate(
          registered = !finished & paper_id %in% (rr_internal_tracking %>% filter(str_detect(type, "Replication")) %>% 
                                                    filter(!is.na(registrations) | prereg_completion == "approve") %>% pull(paper_id))
        ) %>% 
        mutate(
          completed_prereg = !finished & !registered & paper_id %in% (rr_internal_tracking %>% 
                                                                        filter(str_detect(type, "Replication")) %>% filter(prereg_completion == "complete") %>% pull(paper_id))
        ) %>% 
        mutate(
          partial_prereg = !finished & !registered & !completed_prereg & paper_id %in% (rr_internal_tracking %>% 
                                                                                          filter(str_detect(type, "Replication")) %>% filter(prereg_completion == "partial" | osf_activity) %>% pull(paper_id))
        ) %>% 
        mutate(not_started = !papers_not_sourced & !finished & !registered & !completed_prereg & !partial_prereg) %>% 
        pivot_longer(cols = -c(paper_id, journal), names_to = "stage", values_to = "value") %>% 
        group_by(journal, stage) %>% 
        dplyr::summarize(t = sum(value) / n()) %>% 
        ungroup() %>% 
        mutate(journal = as.character(journal)) %>% 
        mutate(
          stage = case_match(
            stage,
            "papers_not_sourced" ~ "Never sourced",
            "not_started" ~ "Not started",
            "partial_prereg" ~ "Partial prereg or OSF",
            "completed_prereg" ~ "Completed prereg",
            "registered" ~ "Registered",
            "finished" ~ "Finished"
          )
        ) %>% 
        mutate(stage = as_factor(stage) %>% fct_relevel(., "Never sourced", "Not started", "Partial prereg or OSF", 
                                                        "Completed prereg", "Registered", "Finished") %>% fct_rev()) %>% 
        group_by(journal) %>% 
        mutate(idx = sum(t[stage == "Never sourced"])) %>% 
        ungroup() %>% 
        ggplot(aes(x = reorder(journal, idx), y = t, fill = stage)) +
        geom_col(alpha = 0.6) +
        scale_fill_manual(values = c("mediumorchid4", "black", "tomato4", "seagreen4", "wheat4", "deepskyblue4")) +
        coord_flip() +
        labs(
          title = "Evidence set (n = 600)",
          x = "",
          y = "",
          fill = ""
        ) +
        theme_light() +
        theme(
          plot.title = element_text(size = 16, hjust = 0.5, face = "bold"),
          axis.text.x = element_text(size = 14),
          axis.text.y = element_text(size = 9),
          legend.text = element_text(size = 14),
          legend.position = "bottom",
          text = element_text(family = "Arial")
        ) +
        #guides(fill = guide_legend(nrow = 6, byrow = T, reverse = T))
        guides(fill = guide_legend(nrow = 1,
                                   byrow = T,
                                   reverse = T,
                                   direction="horizontal"))
      
      figure_s9 <- bundle_ggplot(
        plot = plot,
        width = 4000,height = 4000,units = "px",bg="white")
    }
    
    # Figure S10
    {
      plot <- papers_not_sourced %>% 
        left_join(paper_metadata %>% select(paper_id, field = COS_pub_category), by = "paper_id") %>% 
        select(paper_id, field, dr_reason) %>% 
        mutate(dr_reason = ifelse(is.na(dr_reason), "plausible", dr_reason)) %>% 
        count(field, dr_reason) %>% 
        group_by(field) %>% 
        mutate(prop = n / sum(n)) %>% 
        ungroup() %>% 
        select(-n) %>% 
        mutate(dr_reason = as_factor(dr_reason) %>% 
                 fct_relevel(., "unique event or program", "archival data", "longitudinal", 
                             "sensitive population", "secondary", "plausible")) %>% 
        mutate(
          field = case_match(
            field,
            "economics and finance" ~ "economics",
            "psychology and health" ~ "psychology",
            "sociology and criminology" ~ "sociology",
            .default = field
          )
        ) %>% 
        mutate(field = str_to_sentence(field)) %>% 
        ggplot(aes(x = field, y = prop, fill = dr_reason)) +
        geom_col(alpha = 0.7) +
        scale_fill_manual(values = c("mediumorchid4", "black", "tomato4", "seagreen4", "wheat4", "deepskyblue4"),
                          labels = c("Unique event", "Archival data", "Longitudinal", "Sensitive population",
                                     "Secondary data", "Plausible")) +
        labs(
          x = "",
          y = "",
          fill = "",
          title = "Plausibility of new data replications among unmatched papers (n = 380)"
        ) +
        theme_light() +
        theme(
          plot.title = element_text(size = 12, hjust = 0.5),
          legend.text = element_text(size = 11),
          legend.position = "bottom",
          text = element_text(family = "Arial")
        )
      
      figure_s10 <- bundle_ggplot(
        plot = plot,
        width = 2000,height = 2000,units = "px",bg="white")
    }
    
    # Figure S11
    {
      plot <- papers_not_sourced %>% 
        left_join(paper_metadata %>% select(paper_id, field = COS_pub_category), by = "paper_id") %>% 
        select(paper_id, field, dar_reason) %>% 
        mutate(
          dar_reason = case_match(
            dar_reason,
            "administrative data" ~ "Admin. data",
            "archival data" ~ "Unique data or event",
            "historical event" ~ "Unique data or event",
            "insufficient new cases" ~ "Insufficient new data",
            "own data collection" ~ "Primary data",
            "panel" ~ "Longitudinal",
            "unique data" ~ "Unique data or event",
            "unique event or program" ~ "Unique data or event",
            .default = "Plausible"
          )
        ) %>% 
        count(field, dar_reason) %>% 
        group_by(field) %>% 
        mutate(prop = n / sum(n)) %>% 
        ungroup() %>% 
        select(-n) %>% 
        mutate(dar_reason = as_factor(dar_reason) %>% 
                 fct_relevel(., "Longitudinal", "Admin. data", "Insufficient new data", 
                             "Unique data or event", "Primary data", "Plausible")) %>% 
        mutate(
          field = case_match(
            field,
            "economics and finance" ~ "economics",
            "psychology and health" ~ "psychology",
            "sociology and criminology" ~ "sociology",
            .default = field
          )
        ) %>% 
        mutate(field = str_to_sentence(field)) %>% 
        ggplot(aes(x = field, y = prop, fill = dar_reason)) +
        geom_col(alpha = 0.7) +
        scale_fill_manual(values = c("mediumorchid4", "black", "tomato4", "seagreen4", "wheat4", "deepskyblue4")) +
        labs(
          x = "",
          y = "",
          fill = "",
          title = "Plausibility of secondary data replications among unmatched papers (n = 380)"
        ) +
        theme_light() +
        theme(
          plot.title = element_text(size = 12, hjust = 0.5),
          legend.text = element_text(size = 11),
          legend.position = "bottom",
          text = element_text(family = "Arial")
        )
      figure_s11 <- bundle_ggplot(
        plot = plot,
        width = 2000,height = 2000,units = "px",bg="white")
    }
    
    # Figure S12
    {
      # Data wrangling
      {
        # repli_binary <- repli_outcomes[c("paper_id", "claim_id",
        #                                  df.binvars$binvars.raw)]
        
        # Summarize by proportions w/ CIs
        repli_binary_binaries <- repli_binary[,3:ncol(repli_binary)]
        repli_binary_binaries <- repli_binary_binaries[c(binvars.order)]
        colnames(repli_binary_binaries) <- df.binvars$binvars.full.text
        
        # Generate a results matrix
        repli_binary_cors <- matrix(NA,nrow=ncol(repli_binary_binaries),ncol=ncol(repli_binary_binaries))
        rownames(repli_binary_cors) <- colnames (repli_binary_cors) <- colnames(repli_binary_binaries)
        
        get_wt_corr <- function(i.x,i.y){
          df.cors <- repli_binary_binaries[,c(i.x,i.y)]
          df.cors$paper_id <- repli_binary$paper_id
          df.cors <- na.omit(df.cors)
          df.cors <- df.cors %>%
            group_by(paper_id) %>%
            #mutate(weight = 1/n())
            mutate(weight = 1)
          
          weightedCorr(as.integer(unlist(df.cors[,1])),as.integer(unlist(df.cors[,2])),weights=df.cors$weight,method="Pearson")
        }
        
        # Get pairwise complete weighted correlations
        for(i in 1:nrow(repli_binary_cors)){
          for(j in 1:(i)){
            if(i==j){
              repli_binary_cors[i,j] <- 1
            } else{
              repli_binary_cors[j,i] <- get_wt_corr(i,j)
              repli_binary_cors[i,j] <- NA
            }
          }
        }
        
        binvars.x <- rownames(repli_binary_cors)
        binvars.y <- rownames(repli_binary_cors)
        df.chart <- expand.grid(binvars.x=binvars.x, binvars.y=binvars.y)
        df.chart$corr <-c(repli_binary_cors)
        df.chart$upper_text_label <- str_sub(ifelse(is.na(df.chart$corr) | df.chart$binvars.x==df.chart$binvars.y,"",
                                                    format.round(df.chart$corr,2)), 2, -1)
        df.chart$corrs.all <- ifelse(is.na(df.chart$corr) | df.chart$binvars.x==df.chart$binvars.y,NA,
                                     df.chart$corr)
        
      }
      
      # Chart generation
      {
        chart.palette <- palette_score_charts[7]
        
        top <- ggplot(df.chart, aes(x=binvars.x, y=binvars.y, fill= corr)) + 
          geom_tile()+
          geom_text(aes(x=binvars.y, y=binvars.x, label=upper_text_label),size=3,color="grey30")+
          scale_x_discrete(position = "top") +
          scale_y_discrete(limits=rev)+
          theme_minimal()+
          theme(
            plot.title = element_blank(),
            axis.title = element_blank(),
            axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=0,color="black"),
            axis.text.y = element_text(color="black"),
            legend.key.height= unit(1, 'cm'),
            panel.grid = element_blank(),
            legend.position="none"
          ) +
          scale_fill_gradient(low="white", high=chart.palette[1], limits=c(0, 1),na.value="white")
        
        df.chart$group <- "Distribution of\n all 144 correlations"
        bottom <- ggplot(df.chart, aes(x=corrs.all,y=group,fill=..x..)) + 
          #geom_density(fill=palette_score_charts[6])+
          geom_density_ridges_gradient(linetype=0)+
          scale_x_continuous(limits=c(0,1),breaks=seq(0,1,.1),expand=c(0,0))+
          scale_y_discrete(expand=c(0,0))+
          scale_fill_gradient( limits=c(0, 1),low="white", high=chart.palette[1])+
          geom_vline(xintercept=c(0,1))+
          #ylab("Distribution of\n all 144 correlations")+
          # scale_y_discrete(limits=rev)+
          theme_minimal()+
          theme(
            plot.title = element_blank(),
            axis.title = element_blank(),
            axis.text.y = element_text(angle = 0, vjust = 0, hjust=1 ,size=8),
            axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=0,size=6),
            legend.key.height= unit(1, 'cm'),
            panel.grid = element_blank(),
            legend.position="none",
            text = element_text(family = "Arial")
          )
        
        #bottom_row <- plot_grid(ggplot()+theme_nothing(),figure_3_bottom,ggplot()+theme_nothing(),nrow=1,rel_widths = c(1,2,1))
        #figure_3 <- plot_grid(figure_3_top,bottom_row,ncol=1,rel_heights = c(5,1))
        
        plot <- plot_grid(top,bottom,ncol=1,rel_heights = c(5,1),align = "v")
        
        figure_s12 <- bundle_ggplot(
          plot = plot,
          width = 2000,height = 2000,units = "px",bg="white")
      }
      
    }
    
    # Figure S13
    {
      plot <- repli_outcomes %>%
        filter(!is_covid & repli_version_of_record) %>%
        select(claim_id, report_id, repli_conv_r, repli_score_criteria_met, repli_pattern_criteria_met) %>%
        left_join(orig_outcomes %>% select(claim_id, orig_conv_r), by = "claim_id") %>%
        mutate(orig_conv_r = abs(orig_conv_r)) %>%
        mutate(repli_conv_r = ifelse(repli_pattern_criteria_met, abs(repli_conv_r), -1*abs(repli_conv_r))) %>%
        mutate(diff = repli_conv_r - orig_conv_r) %>%
        select(report_id, diff, score = repli_score_criteria_met) %>%
        left_join(repli_binary, by = "report_id") %>%
        rename_with(., .fn = function(x) str_remove_all(x, "repli_binary_")) %>%
        # pivot_longer(cols = -c("report_id", "diff"), names_to = "metric", values_to = "outcome") %>%
        pivot_longer(cols = -c("report_id", "diff","paper_id","claim_id"), names_to = "metric", values_to = "outcome") %>%
        mutate(outcome = as.numeric(outcome)) %>%
        drop_na() %>%
        mutate(
          metric = case_match(
            metric,
            "analyst" ~ "Analyst interpretation",
            "repli_score_criteria_met" ~ "Sig + pattern",
            "score" ~ "Sig + pattern",
            "sum_p" ~ "Sum of p-values",
            "skep_p" ~ "Skeptical p-value",
            "orig_wthn" ~ "Orig. in rep. CI",
            "rep_wthn" ~ "Rep. in orig. CI",
            "wthn_pi" ~ "Rep. in prediction interval",
            "meta_success" ~ "Meta-analysis",
            "bma_result" ~ "Bayesian meta-analysis",
            "telescopes" ~ "Small telescopes",
            "bf_result" ~ "Bayes factor",
            "bayes_rep" ~ "Replication Bayes factor",
            "correspondence" ~ "Correspondence test",
          )
        ) %>%
        ggplot(aes(x = diff, y = outcome)) +
        geom_point(alpha = 0.35) +
        scale_y_continuous(breaks = c(0, 1), labels = c("Fail", "Success")) +
        stat_smooth(method = "glm", method.args = list(family = binomial), se = F, color = "deepskyblue3") +
        labs(
          x = "Difference in effect size (replication - original)",
          y = ""
        ) +
        facet_wrap(~metric) +
        theme_light() +
        theme(
          strip.background = element_blank(),
          strip.text = element_text(color = "black"),
          text = element_text(family = "Arial")
        )
      
      figure_s13 <- bundle_ggplot(
        plot = plot,
        width = 2000,height = 2000,units = "px",bg="white")
    }
    
    # Figure S14
    {
      # Data wrangling
      {
        
        # Summarize by proportions w/ CIs
        binary.proportions <- 
          do.call(rbind,lapply(df.binvars$binvars.raw,function(binary.var){
            repli_binary_temp <- na.omit(repli_binary[c("paper_id",binary.var)])
            repli_binary_temp <- repli_binary_temp %>%
              group_by(paper_id) %>%
              mutate(weight = 1)
              # mutate(weight = 1/n())
            
            n_testable_claims <- nrow(repli_binary_temp)
            p_testable_claims <- n_testable_claims/length(unique(repli_outcomes$claim_id))
            n_passed_claims <- sum(repli_binary_temp[[binary.var]])
            p_passed_temp <- binconf(n_passed_claims,n_testable_claims)
            p_passed_claims <- p_passed_temp[1]
            p_passed_CI_LB_claims <- p_passed_temp[2]
            p_passed_CI_UB_claims <- p_passed_temp[3]
            
            n_testable_papers <- length(unique((repli_binary_temp$paper_id)))
            p_testable_papers <- n_testable_papers/length(unique(repli_outcomes$paper_id))
            n_passed_papers <- sum(repli_binary_temp[[binary.var]]*repli_binary_temp$weight)
            p_passed_temp <- cw.proportion(as.integer(repli_binary_temp[[binary.var]]),
                                           repli_binary_temp$weight,repli_binary_temp$paper_id,iters)
            p_passed_papers <- p_passed_temp$point.estimate
            p_passed_CI_LB_papers <- p_passed_temp$CI.lb
            p_passed_CI_UB_papers <- p_passed_temp$CI.ub
            
            data.frame(binary.var,
                       n_testable_claims,p_testable_claims,n_passed_claims,p_passed_claims,p_passed_CI_LB_claims,p_passed_CI_UB_claims,
                       n_testable_papers,p_testable_papers,n_passed_papers,p_passed_papers,p_passed_CI_LB_papers,p_passed_CI_UB_papers
            )
          }))
        
        binary.proportions$binary.var <- ordered(binary.proportions$binary.var,
                                                 levels = binvars.raw,
                                                 labels= binvars.full.text)
        
        
        # Reorder by size
        binary.proportions <- binary.proportions[order(-binary.proportions$p_passed_papers),]
        binary.proportions$binary.var <- ordered(as.character(binary.proportions$binary.var),
                                                 levels = binary.proportions$binary.var,
                                                 labels= binary.proportions$binary.var)
        
        binary.proportions$num <- rev(1:nrow(binary.proportions))
        
        binary.proportions$n.text <- paste0("n=",format.round(binary.proportions$n_passed_papers,1),"/",binary.proportions$n_testable_papers)
        
        binary.proportions$num.claims <- rev(1:nrow(binary.proportions))
        
        binary.proportions$n.text.claims <- paste0("n=",format.round(binary.proportions$n_passed_claims,0),"/",binary.proportions$n_testable_claims)
      }
        
      # Chart generation
      {
        plot <- ggplot(binary.proportions, aes(x=reorder(binary.var, -p_passed_claims),y=p_passed_claims)) + 
          geom_bar(stat="identity",fill="grey90") +
          geom_text(aes(x=num.claims,y=0.02,label=n.text.claims),hjust=0,size=3)+
          funkyheatmap::geom_rounded_rect(
            aes(xmin=num-.45,xmax=num+.45,ymin = p_passed_CI_LB_claims, ymax = p_passed_CI_UB_claims),
            radius=unit(3, "points"),show.legend=FALSE,color="black",size=0,fill=palette_score_charts[7])+
          geom_segment(aes(x=num.claims-.45,xend=num.claims+.45,y = p_passed_claims, yend = p_passed_claims),color="white")+
          # geom_errorbar(aes(y=PointEst,ymin=Lower, ymax=Upper), width=.2)+
          theme_minimal()+
          scale_x_discrete(limits=rev(binary.proportions$binary.var))+
          coord_flip()+
          scale_y_continuous(expand=expansion(add = c(0, .05)),
                             labels = scales::percent_format(),
                             breaks=c(0,.25,.5,.50,1),
                             limits=c(0,1))+
          theme(legend.position = "none",
                legend.title=element_blank(),
                panel.border = element_blank(),
                panel.grid = element_blank(),
                axis.line = element_blank(),
                text = element_text(family = "Arial")
          )+
          xlab("Binary assessments")+
          ylab("Percentage of claims replicated successfully")
        
        figure_s14 <- bundle_ggplot(
          plot = plot,
          width = 3000,height = 1000,units = "px",bg="white")
      }
    }
    
    # Figure S15
    {
      p <- ggplot(llm_method_data, aes(reorder(llm_category2, -llm_score), llm_score, fill = llm_score)) +
        #scale_fill_viridis_c(option = "viridis", begin = 0.0, end = 0.9) +
        geom_bar(stat = "identity", position = position_dodge(width = 0.55), alpha = 1,fill=palette_score_charts[7]) +
        geom_errorbar(aes(ymin = llm_score_lb, ymax = llm_score_ub), width = 0.2, linewidth = 0.4, color = "black", position = position_dodge(width = 0.55)) +
        scale_y_continuous(breaks = seq(0, 100, 10),expand=c(0,0)) +
        scale_x_discrete(expand=c(0,0)) +
        #scale_alpha_manual(values = c(1, 0.4)) +
        coord_cartesian(ylim = c(0, 100)) +
        labs(x = "", y = "Papers (%) using method/technique") +
        theme_light()+
        theme(
          axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 10),
          axis.text.y = element_text(size = 12),
          axis.title.y = element_text(size = 12),
          legend.position = "none",
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          text = element_text(family = "Arial")
        )
      
      figure_s15 <- bundle_ggplot(
        plot = p,
        width = 3000,height = 2000,units = "px",bg="white")
    }
    
    # Figure S16
    {
      p <- ggplot(llm_theory_data, aes(reorder(llm_category2, -llm_score), llm_score, fill = llm_score)) +
        #scale_fill_viridis_c(option = "viridis", begin = 0.0, end = 0.9) +
        geom_bar(stat = "identity", position = position_dodge(width = 0.55), alpha = 1,fill=palette_score_charts[7]) +
        geom_errorbar(aes(ymin = llm_score_lb, ymax = llm_score_ub), width = 0.2, linewidth = 0.4, color = "black", position = position_dodge(width = 0.55)) +
        scale_y_continuous(breaks = seq(0, 60, 10),expand=c(0,0)) +
        scale_x_discrete(expand=c(0,0)) +
        scale_alpha_manual(values = c(1, 0.4)) +
        coord_cartesian(ylim = c(0, 60)) +
        labs(x = "", y = "Papers (%) citing framework/paradigm") +
        theme_light()+
        theme(
          axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 10),
          axis.text.y = element_text(size = 12),
          axis.title.y = element_text(size = 12),
          legend.position = "none",
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          text = element_text(family = "Arial")
        )
      
      figure_s16 <- bundle_ggplot(
        plot = p,
        width = 3000,height = 2000,units = "px",bg="white")
    }
  }
  
  # Export
  {
    return(list(
      "figure_1"=figure_1,
      "figure_2"=figure_2,
      "figure_3"=figure_3,
      "figure_4"=figure_4,
      "figure_s2"=figure_s2,
      "figure_s3"=figure_s3,
      "figure_s4"=figure_s4,
      "figure_s5"=figure_s5,
      "figure_s6"=figure_s6,
      "figure_s7"=figure_s7,
      "figure_s8"=figure_s8,
      "figure_s9"=figure_s9,
      "figure_s10"=figure_s10,
      "figure_s11"=figure_s11,
      "figure_s12"=figure_s12,
      "figure_s13"=figure_s13,
      "figure_s14"=figure_s14,
      "figure_s15"=figure_s15,
      "figure_s16"=figure_s16
      ))
  }
}


