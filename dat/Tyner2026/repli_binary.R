calculate_repli_binary <- function(repli_outcomes, 
                                   orig_outcomes, 
                                   paper_metadata) {
  
  # Sub-functions ----
  
  # meta-analyze the original and replication effects
  meta <- function(x) {
    
    es <- escalc(
      measure = "GEN",
      yi = c(x[1], x[2]),
      sei = c(x[3], x[4])
    )
    
    mod <- rma(yi = yi, vi = vi, data = es, method = "FE")
    
    c(mod$b, mod$ci.lb, mod$ci.ub) %>% 
      as.character() %>% 
      str_c(., collapse = ", ")
    
  }
  
  diff_test <- function(tau1, tau2, se1, se2, alpha = .05) {
    
    # Conducts the difference test
    # tau1, tau2 ... numeric of length 1, effect estimates from study 1 and 2
    # se1, se2   ... numeric of length 1, the standard error from study 1 and 2
    # alpha      ... numeric of length 1, desired alpha level for the test
    
    # OUTPUT: Returns test significance, point estimate of difference, 
    #         confidence intervals, and p-value
    
    se <- sqrt(se1^2 + se2^2) # assumes independence
    
    delta <- abs(tau2 - tau1)
    
    z <- delta / se
    
    p <- pnorm(abs(z), lower.tail = FALSE)
    
    out1 <- p <= alpha / 2 # check for significance
    
    z_crit <- qnorm(alpha / 2, lower.tail = FALSE) # critical value (quantile)
    
    ci1 <- tau1 + c(-z_crit, z_crit) * se1 # CI for Study 1
    
    ci2 <- tau2 + c(-z_crit, z_crit) * se2 # CI for Study 2
    
    diff_ci <- delta + c(-z_crit, z_crit) * se # CI for the difference
    
    list("Significance" = out1, "Difference" = delta, 
         "CI_tau1" =  ci1, "CI_tau2" = ci2,
         "CI_difference" = diff_ci, "pval" = 2 * p)
  }
  
  equiv_test <- function(tau1, tau2, se1, se2, e_th, alpha = .05) {
    
    # Conducts the equivalence test
    # tau1, tau2 ... numeric of length 1, effect estimates from study 1 and 2
    # se1, se2   ... numeric of length 1, standard error from study 1 and 2
    # alpha      ... numeric of length 1, desired alpha level for the test
    # e_th       ... numeric of length 1, equivalence threshold
    
    # OUTPUT: Returns significance of test, point estimate of difference, 
    #         confidence intervals, and p-values for both one-sided tests
    
    se <- sqrt(se1^2 + se2^2)
    
    delta <- tau1 - tau2
    
    z_eq <- (delta + c(-e_th, e_th)) / se # upper & lower test statistics
    
    p_eqp <- pnorm(z_eq[1], lower.tail = TRUE)
    p_eqn <- pnorm(z_eq[2], lower.tail = FALSE)
    p_eq <- c(p_eqp, p_eqn)
    
    out2 <- all(p_eq <= alpha)                   
    
    z_crit <- qnorm(alpha, lower.tail = FALSE)
    
    ci1 <- tau1 + c(-z_crit, z_crit) * se1 # CI for Study 1
    
    ci2 <- tau2 + c(-z_crit, z_crit) * se2 # CI for Study 2
    
    list("Significance" = out2, 
         "Difference" = delta, 
         "CI_tau1" =  ci1, 
         "CI_tau2" = ci2, 
         "pval_below_e_th" = p_eq[1], 
         "pval_above_e_th" = p_eq[2])
  }
  
  corresp_test <- function(tau1, tau2, se1, se2, e_th, alpha = .05) {
    
    # Conducts the correspondence test
    # tau1, tau2 ... numeric of length 1, effect estimates from study 1 and 2
    # se1, se2   ... numeric of length 1, standard error from study 1 and 2
    # e_th       ... numeric of length 1, equivalence threshold
    # alpha      ... numeric of length 1, desired alpha level for the test
    
    # OUTPUT: Returns the output from diff_test and equiv_test functions, and
    #         result of the correspondence test
    
    out1 <- diff_test(tau1, tau2, se1, se2, alpha)
    
    out2 <- equiv_test(tau1, tau2, se1, se2, e_th, alpha)
    
    # determine replication success
    res <- ifelse(out1[[1]], 
                  ifelse(out2[[1]], "Trivial Difference", "Difference"), 
                  ifelse(out2[[1]], "Equivalence", "Indeterminacy"))
    
    # difference
    diff_out <- c(out1[[2]], out1[[3]], out1[[4]], out1[[5]], out1[[6]])
    names(diff_out) <- c("Difference", 
                         "tau1_Lower_CI", "tau1_Upper_CI", 
                         "tau2_Lower_CI", "tau2_Upper_CI", 
                         "diff_Lower_CI", "diff_Upper_CI",
                         "p")
    # equivalence
    equiv_out <- c(out2[[2]], out2[[3]], out2[[4]], out2[[5]], out2[[6]])
    names(equiv_out) <- c("Difference", 
                          "tau1_Lower_CI", "tau1_Upper_CI", 
                          "tau2_Lower_CI", "tau2_Upper_CI", 
                          "p_greater", "p_less")
    
    # final output
    list("Difference" = diff_out, 
         "Equivalence" = equiv_out, 
         "Correspondence" = res)
  }
  
  ## small telescopes
  # what effect size would give the original study 33% power?
  get_small <- function(x) {
    
    # NA if original sample size is missing or if there are <4 observations
    if (is.na(x) | x < 4) {
      NA_real_
    } else {
      pwr.r.test(n = x, power = 1/3)$r
    }
    
  }
  
  ## bayesian meta-analysis
  custom_bma <- function(x) {
    
    try_meta <- try(meta_fixed(c(x[1], x[2]), c(x[3], x[4]), 
                               d = prior("norm", c(mean = 0, sd = .25)),
                               iter = 1500, 
                               rel.tol = .1, 
                               logml = "integrate", 
                               summarize = "integrate") %>% 
                      pluck("BF") %>% 
                      as.data.frame() %>% 
                      tibble::rownames_to_column() %>% 
                      as_tibble() %>% 
                      filter(rowname == "fixed_H1") %>% 
                      pull(fixed_H0))
    
    if("try-error" %in% class(try_meta)) {
      
      NA_real_
      
    } else {
      
      meta_fixed(c(x[1], x[2]), c(x[3], x[4]), 
                 d = prior("norm", c(mean = 0, sd = .25)),
                 iter = 1500, 
                 rel.tol = .1, 
                 logml = "integrate", 
                 summarize = "integrate") %>% 
        pluck("BF") %>% 
        as.data.frame() %>% 
        tibble::rownames_to_column() %>% 
        as_tibble() %>% 
        filter(rowname == "fixed_H1") %>% 
        pull(fixed_H0)
      
    }
    
  }
  
  # Calculations ----
  
  # Remove COVID cases
  # Use version of record so it's just one outcome per claim
  repli <- repli_outcomes %>% 
    filter(!is_covid & repli_version_of_record) %>% 
    left_join(paper_metadata %>% 
                select(paper_id, COS_pub_category), 
              by = "paper_id") %>% 
    rename(field = COS_pub_category) %>% 
    mutate(field = str_to_sentence(field))
  
  orig <- orig_outcomes %>% 
    semi_join(repli, by = "claim_id") %>% 
    left_join(paper_metadata %>% 
                select(paper_id, COS_pub_category), 
              by = "paper_id") %>% 
    rename(field = COS_pub_category) %>% 
    mutate(field = str_to_sentence(field))
  
  # pattern and significance
  score <- repli %>% 
    select(report_id, field, repli_score_criteria_met)
  
  # analyst interpretation
  # convert complicated to NA for the purposes of this summary
  # drop 'complicated' cases
  interpret <- repli %>% 
    select(report_id, field, analyst = repli_interpret_supported) %>% 
    filter(analyst != "complicated") %>% 
    mutate(analyst = analyst == "yes")
  
  # is original within repli CI
  orig_within <- repli %>% 
    select(report_id, claim_id, field, repli_conv_r_lb, repli_conv_r_ub) %>% 
    left_join(orig %>% 
                select(claim_id, orig_conv_r), 
              by = "claim_id") %>% 
    mutate(orig_wthn = between(orig_conv_r, 
                               repli_conv_r_lb, 
                               repli_conv_r_ub)) %>% 
    drop_na(orig_wthn)
  
  # is replication within orig CI
  rep_within <- repli %>% 
    select(report_id, claim_id, field, repli_conv_r) %>% 
    left_join(orig %>% 
                select(claim_id, orig_conv_r_lb, orig_conv_r_ub), 
              by = "claim_id") %>% 
    mutate(rep_wthn = between(repli_conv_r, orig_conv_r_lb, orig_conv_r_ub)) %>% 
    drop_na(rep_wthn)
  
  # significance and direction of meta-analytic CI
  # need to estimate a standard error for each of the effects
  # default to the effective df or sample size whenever it's available
  repli_meta <- repli %>% 
    mutate(repli_df1 = ifelse(is.na(repli_effective_df1), 
                              repli_stat_dof_1, 
                              repli_effective_df1)) %>% 
    mutate(repli_size = ifelse(is.na(repli_effective_sample_size), 
                               repli_sample_size_value, 
                               repli_effective_sample_size)) %>% 
    mutate(repli_se = ifelse(repli_stat_type == "t",
                             sqrt((1 - repli_conv_r^2)/repli_df1),
                             sqrt((1 - repli_conv_r^2)/repli_size)))
  
  orig_meta <- orig %>% 
    mutate(orig_df1 = ifelse(is.na(original_effective_df1_reference),
                             orig_stat_dof_1, 
                             original_effective_df1_reference)) %>% 
    mutate(orig_size = ifelse(is.na(original_effective_sample_size),
                              orig_sample_size_value, 
                              original_effective_sample_size)) %>% 
    mutate(orig_se = ifelse(orig_stat_type == "t",
                            sqrt((1 - orig_conv_r^2)/orig_df1),
                            sqrt((1 - orig_conv_r^2)/orig_size)))
  
  combined_meta <- repli_meta %>% 
    select(report_id, paper_id, claim_id, field, repli_conv_r, repli_se) %>% 
    left_join(orig_meta %>% 
                select(claim_id, orig_conv_r, orig_se), 
              by = "claim_id") %>% 
    drop_na(repli_se) %>% 
    drop_na(orig_se) %>% 
    mutate(es = select(., c(repli_conv_r, orig_conv_r, repli_se, orig_se)) %>% 
             apply(1, meta)) %>% 
    separate(es, into = c("b", "lb_b", "ub_b"), sep = ", ") %>% 
    mutate(across(.cols = c(b, lb_b, ub_b), as.numeric))
  
  # success means meta CI doesn't cross 0 and it has the same direction as the
  # original effect
  metaanalytic <- combined_meta %>% 
    mutate(sig = sign(lb_b) == sign(ub_b)) %>% 
    mutate(dir = sign(orig_conv_r) == sign(lb_b)) %>% 
    mutate(meta_success = sig & dir) 
  
  # repli within original prediction interval
  # defining prediction intervals as done in https://osf.io/z4gjn:
  # comparison_es_lb_pi_nativeunits <- es_o - se_comb*qnorm(.975)
  # comparison_es_ub_pi_nativeunits <- es_o + se_comb*qnorm(.975)
  # NOTE: these can go above 1 and below -1, 
  # so just manually capping them for now
  prediction_interval <- combined_meta %>% 
    mutate(se_comb = select(., c(orig_se, repli_se)) %>% 
             apply(1, function(x) sqrt((x[1])^2 + (x[2])^2))) %>% 
    mutate(o_lb_pi = select(., c(orig_conv_r, se_comb)) %>% 
             apply(1, function(x) x[1] - (x[2])*qnorm(.975)),
           o_ub_pi = select(., c(orig_conv_r, se_comb)) %>% 
             apply(1, function(x) x[1] + (x[2])*qnorm(.975))) %>% 
    mutate(o_lb_pi = ifelse(o_lb_pi < -1, -1, o_lb_pi),
           o_ub_pi = ifelse(o_ub_pi > 1, 1, o_ub_pi)) %>% 
    mutate(wthn_pi = between(repli_conv_r, o_lb_pi, o_ub_pi))
  
  # sum of p-values 
  # per the article, the sum of the two *one-tailed* p-values should be 
  # less than 0.035
  # can only use exact p-values from orig
  total_pvalue <- repli %>% 
    left_join(
      orig %>% 
        select(claim_id, orig_p_value, orig_p_value_type, orig_p_value_tails),
      by = "claim_id"
    ) %>% 
    filter(orig_p_value_type == "exact") %>% 
    mutate(orig_p_value = ifelse(is.na(orig_p_value_tails) | 
                                   str_detect(orig_p_value_tails, 
                                              "two"),
                                 orig_p_value,
                                 orig_p_value*2)) %>% 
    mutate(across(c(orig_p_value, repli_p_value),
                  function(x) x/2)) %>% 
    mutate(both_p = (orig_p_value + repli_p_value) < 0.035) %>% 
    mutate(sum_p = both_p & repli_pattern_criteria_met)
  
  # adjust sign of the small effect to match the original finding
  # TRUE means that original design could have detected a small effect
  small_telescopes <- orig %>% 
    select(claim_id, field, orig_conv_r, orig_sample_size_value) %>% 
    mutate(small_effect = select(., orig_sample_size_value) %>% 
             apply(1, get_small)) %>% 
    mutate(small_effect = ifelse(sign(orig_conv_r) == 1, 
                                 small_effect, 
                                 (-1)*small_effect)) %>% 
    left_join(
      repli %>% 
        select(claim_id, report_id, repli_conv_r_lb, repli_conv_r_ub),
      by = "claim_id"
    ) %>% 
    mutate(telescopes = ifelse(sign(small_effect) == 1, 
                               repli_conv_r_ub > small_effect, 
                               small_effect > repli_conv_r_lb)) %>% 
    drop_na(telescopes)
  
  ## bayes factor
  # F stat
  F_bf <- repli %>% 
    filter(repli_stat_type == "F") %>% 
    select(report_id, 
           repli_sample_size_value, 
           repli_stat_value, 
           repli_stat_dof_1, 
           repli_stat_dof_2) %>% 
    mutate(bf = select(., 
                       c(repli_stat_value, 
                         repli_sample_size_value, 
                         repli_stat_dof_1, 
                         repli_stat_dof_2)) %>% 
             apply(1, 
                   function(x) f_test_BFF(x[1], x[2], x[3], x[4]) %>% 
                     pluck("log_bf"))
    )
  
  # z score
  # go with effective sample size in cases where it exists
  z_bf <- repli %>% 
    filter(repli_stat_type == "z") %>% 
    select(report_id, 
           repli_stat_value, 
           repli_sample_size_value, 
           repli_effective_sample_size) %>% 
    mutate(samp_size = select(., 
                              c(repli_sample_size_value, 
                                repli_effective_sample_size)) %>% 
             apply(1, min, na.rm = T)) %>% 
    mutate(bf = select(., c(repli_stat_value, samp_size)) %>% 
             apply(1, 
                   function(x) z_test_BFF(x[1], n = x[2], one_sample = T) %>%
                     pluck("log_bf")))
  
  # for t, only include cases with degrees of freedom
  # also use BayesFactor package rather than BFF
  t_bf <- repli %>% 
    filter(repli_stat_type == "t") %>% 
    filter(!is.na(repli_stat_dof_1)) %>% 
    select(report_id, repli_stat_value, repli_sample_size_value) %>% 
    mutate(bf = select(., c(repli_stat_value, repli_sample_size_value)) %>% 
             apply(1, 
                   function(x) ttest.tstat(x[1], n1 = x[2]) %>% 
                     pluck("bf")))
  
  # for chi-square, seems like the most straightforward approach is to use the 
  # converted partial correlations
  # this gets around the uncertainty about the log scale with the chi-squared 
  # approach
  # SE formula: sqrt((1 - r^2)/df) [use sample size for the df value]
  
  # maybe verify the se forumula for the rest as well
  chi_bf <- repli %>% 
    filter(repli_stat_type == "chi_squared") %>% 
    mutate(se = select(., c(repli_conv_r, repli_sample_size_value)) %>% 
             apply(1, 
                   function(x) sqrt((1 - (x[1])^2)/x[2]))) %>% 
    select(report_id, repli_conv_r, repli_sample_size_value, se) %>% 
    mutate(t = repli_conv_r/se) %>% 
    mutate(bf = select(., c(t, repli_sample_size_value)) %>% 
             apply(1, 
                   function(x) ttest.tstat(x[1], n1 = x[2]) %>% 
                     pluck("bf")))
  
  all_bf <- t_bf %>% select(report_id, bf) %>% 
    bind_rows(F_bf %>% select(report_id, bf)) %>% 
    bind_rows(z_bf %>% select(report_id, bf)) %>% 
    bind_rows(chi_bf %>% select(report_id, bf))
  
  bayes_factor <- all_bf %>% 
    mutate(bf_outcome = interpret_bf(bf, rules = "jeffreys1961", log = T)) %>% 
    mutate(bf_interpret = str_detect(bf_outcome, "favour") & 
             (str_detect(bf_outcome, "strong") | 
                str_detect(bf_outcome, "extreme"))) %>% 
    left_join(repli %>% 
                select(report_id, field, repli_pattern_criteria_met), 
              by = "report_id") %>% 
    mutate(bf_result = bf_interpret & repli_pattern_criteria_met)
  
  ## replication bayes factor
  rep_bayes <- combined_meta %>% 
    mutate(r_bf = select(., 
                         c(orig_conv_r, orig_se, repli_conv_r, repli_se)) %>% 
             apply(1, 
                   function(x) BFr(x[1], x[2], x[3], x[4]))) %>% 
    left_join(repli %>% select(report_id, repli_pattern_criteria_met), 
              by = "report_id") %>% 
    mutate(bayes_rep = repli_pattern_criteria_met & r_bf < 1)
  
  bma_estimation <- combined_meta %>% 
    select(report_id, orig_conv_r, repli_conv_r, orig_se, repli_se) %>% 
    mutate(
      meta_bf = select(., orig_conv_r, repli_conv_r, orig_se, repli_se) %>% 
        apply(1, custom_bma)
    )
  
  meta_bma <- bma_estimation %>% 
    mutate(bma_outcome = interpret_bf(meta_bf)) %>% 
    left_join(repli_outcomes %>% select(report_id, repli_pattern_criteria_met), 
              by = "report_id") %>% 
    mutate(bma_interpret = str_detect(bma_outcome, "favour") & 
             (str_detect(bma_outcome, "strong") | 
                str_detect(bma_outcome, "moderate") | 
                str_detect(bma_outcome, "extreme"))) %>% 
    mutate(bma_result = repli_pattern_criteria_met & bma_interpret)
  
  ## skeptical p-value
  # should look at effective sample size as well
  # main priority is confirming the sample size units are the same
  # can't have p-values of 0
  skeptical <- repli %>% 
    select(report_id, 
           claim_id, 
           repli_score_criteria_met, 
           field,
           repli_pattern_criteria_met, 
           repli_sample_size_value, 
           repli_p_value) %>% 
    left_join(orig %>% 
                select(claim_id, 
                       orig_sample_size_value, 
                       orig_p_value, 
                       orig_p_value, 
                       orig_p_value_type, 
                       orig_p_value_tails),
              by = "claim_id") %>% 
    select(-claim_id) %>% 
    filter(orig_p_value_type == "exact") %>% 
    mutate(orig_p_value = ifelse(is.na(orig_p_value_tails) | 
                                   str_detect(orig_p_value_tails, 
                                              "two"),
                                 orig_p_value,orig_p_value*2)) %>% 
    select(-orig_p_value_type, -orig_p_value_tails) %>% 
    mutate(
      across(
        c(repli_p_value, orig_p_value),
        function(x) ifelse(x == 0, 0.0001, x)
      )
    ) %>% 
    mutate(repli_z = p2z(repli_p_value)) %>% 
    mutate(orig_z = p2z(orig_p_value)) %>% 
    mutate(c_ratio = repli_sample_size_value/orig_sample_size_value) %>% 
    drop_na(repli_z, orig_z, c_ratio) %>% 
    mutate(
      revised_p = select(., c(orig_z, repli_z, c_ratio)) %>% 
        apply(1, 
              function(x) pSceptical(zo = x[1], 
                                     zr = x[2], 
                                     c = x[3], 
                                     alternative = "two.sided", 
                                     type = "golden"))
    ) %>% 
    mutate(skep_level = levelSceptical(level = 0.05, 
                                       alternative = "two.sided", 
                                       type = "golden")) %>% 
    mutate(p_criteria = revised_p < skep_level) %>% 
    mutate(skep_p = p_criteria & repli_pattern_criteria_met)
  
  ## correspondence test
  # selecting a moderate threshold of 0.2 -- most theoretically defensible since 
  # it represents a shift in corr. interpretation
  # making it binary by treating 'trivial' as equivalent and indeterminate as NA
  correspondence_test <- combined_meta %>% 
    mutate(
      ct = select(., c(orig_conv_r, repli_conv_r, orig_se, repli_se)) %>% 
        apply(1, function(x) corresp_test(x[1], x[2], x[3], x[4], 
                                          e_th = 0.2, alpha = 0.05) %>% 
                pluck("Correspondence"))
    ) %>% 
    mutate(
      correspondence = case_when(
        ct %in% c("Equivalence", "Trivial Difference") ~ T,
        ct == "Difference" ~ F,
        ct == "Indeterminacy" ~ NA
      )
    )
  
  score %>% 
    left_join(interpret %>% select(report_id, analyst), 
              by = "report_id") %>% 
    left_join(orig_within %>% select(report_id, orig_wthn), 
              by = "report_id") %>% 
    left_join(rep_within %>% select(report_id, rep_wthn), 
              by = "report_id") %>% 
    left_join(metaanalytic %>% select(report_id, meta_success), 
              by = "report_id") %>% 
    left_join(prediction_interval %>% select(report_id, wthn_pi), 
              by = "report_id") %>% 
    left_join(total_pvalue %>% select(report_id, sum_p), 
              by = "report_id") %>% 
    left_join(small_telescopes %>% select(report_id, telescopes), 
              by = "report_id") %>% 
    left_join(rep_bayes %>% select(report_id, bayes_rep), 
              by = "report_id") %>% 
    left_join(meta_bma %>% select(report_id, bma_result), 
              by = "report_id") %>% 
    left_join(skeptical %>% select(report_id, skep_p), 
              by = "report_id") %>% 
    left_join(correspondence_test %>% select(report_id, correspondence), 
              by = "report_id") %>%
    left_join(bayes_factor %>% select(report_id, bf_result), 
              by = "report_id") %>%
    select(report_id,
           repli_binary_analyst = analyst,
           repli_binary_orig_wthn = orig_wthn,
           repli_binary_rep_wthn = rep_wthn,
           repli_binary_meta_success = meta_success,
           repli_binary_wthn_pi = wthn_pi,
           repli_binary_sum_p = sum_p,
           repli_binary_telescopes = telescopes,
           repli_binary_bayes_rep = bayes_rep,
           repli_binary_bma_result = bma_result,
           repli_binary_skep_p = skep_p,
           repli_binary_correspondence = correspondence,
           repli_binary_bf_result = bf_result)
  
}
