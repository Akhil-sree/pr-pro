# =======================================================================
# CATSSR-BN -- exact reproduction using bnlearn (the R package the paper
# actually used), following Zhou et al. (2022), Safety Science 157, 105942.
#
# Run build_binary_matrix.py FIRST to produce bn_matrix.csv + node_family.csv
# from your ASRS export, then run this script.
#
# This script does, in order, exactly what the paper describes:
#   1. Load the binary incident x event matrix, convert to factors
#      (bnlearn requires discrete/factor variables).
#   2. Build a BLACKLIST forbidding any edge between two result-event
#      nodes (paper: "result events can only be caused by causal factor
#      events, and there are no directed edges between result events").
#   3. Structure learning with mmhc() -- the actual Max-Min Hill-Climbing
#      hybrid algorithm from bnlearn, exactly as the paper used.
#   4. Parameter learning with bn.fit(..., method = "bayes") -- Bayesian
#      (BDeu-style) estimation of every node's CPT.
#   5. k-fold cross-validation with bn.cv(..., loss = "logl") -- the
#      log-likelihood loss described in the paper's Eq. 21, k = 10 by
#      default matching the paper.
#   6. Mutual information between every causal factor event and every
#      result event (Eq. 22), reproducing the paper's Table 3 / Table 4
#      style ranking.
#   7. A network diagram (igraph), analogous to the paper's Fig. 1 Gephi
#      plot: node size = degree, colored by causal vs result family.
#
# Usage (local R, or inside Colab's R runtime):
#   install.packages(c("bnlearn", "igraph"))
#   Rscript catssr_bn.R
# =======================================================================

if (!requireNamespace("bnlearn", quietly = TRUE)) install.packages("bnlearn")
if (!requireNamespace("igraph", quietly = TRUE)) install.packages("igraph")

library(bnlearn)
library(igraph)

cat(strrep("=", 70), "\n")
cat("STEP 1: Loading binary matrix\n")
cat(strrep("=", 70), "\n")

data <- read.csv("bn_matrix.csv", check.names = FALSE)
node_family_df <- read.csv("node_family.csv", check.names = FALSE)
node_family <- setNames(node_family_df$family, node_family_df$node)

# bnlearn requires discrete data as factors with explicit levels (0/1),
# even for columns that happen to be all-0 or all-1 in this sample.
for (col in names(data)) {
  data[[col]] <- factor(data[[col]], levels = c(0, 1))
}

cat(sprintf("Loaded %d incidents x %d nodes.\n", nrow(data), ncol(data)))

result_nodes <- names(node_family)[node_family == "result"]
causal_nodes <- names(node_family)[node_family == "causal"]
cat(sprintf("%d causal-factor nodes, %d result-event nodes.\n",
            length(causal_nodes), length(result_nodes)))


# =======================================================================
# STEP 2: Blacklist -- no edges between two result events (either direction)
# =======================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("STEP 2: Building blacklist (no result<->result edges)\n")
cat(strrep("=", 70), "\n")

if (length(result_nodes) > 1) {
  bl <- expand.grid(from = result_nodes, to = result_nodes, stringsAsFactors = FALSE)
  bl <- bl[bl$from != bl$to, ]
} else {
  bl <- data.frame(from = character(0), to = character(0))
}
cat(sprintf("Blacklisted %d directed result<->result edges.\n", nrow(bl)))


# =======================================================================
# STEP 3: Structure learning -- real MMHC (Tsamardinos et al., 2006),
# exactly as the paper specifies, with the blacklist applied.
# =======================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("STEP 3: Structure learning (MMHC, bnlearn native implementation)\n")
cat(strrep("=", 70), "\n")

t0 <- Sys.time()
bn_structure <- mmhc(
  data, blacklist = bl,
  maximize.args = list(score = "bde", iss = 10)
)
t1 <- Sys.time()
cat(sprintf("MMHC structure learning finished in %.1f seconds.\n",
            as.numeric(difftime(t1, t0, units = "secs"))))

arcs_df <- arcs(bn_structure)
cat(sprintf("Learned %d edges across %d nodes.\n", nrow(arcs_df), length(nodes(bn_structure))))
cat("\nLearned edges (parent/independent -> child/dependent):\n")
for (i in seq_len(nrow(arcs_df))) {
  cat(sprintf("  %-8s -> %s\n", arcs_df[i, "from"], arcs_df[i, "to"]))
}

write.csv(arcs_df, "bn_edges.csv", row.names = FALSE)
cat("\nSaved bn_edges.csv\n")


# =======================================================================
# STEP 4: Parameter learning -- Bayesian (BDeu-style) estimation of CPTs
# =======================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("STEP 4: Fitting CPTs (Bayesian/BDeu estimation)\n")
cat(strrep("=", 70), "\n")

fitted_bn <- bn.fit(bn_structure, data, method = "bayes", iss = 10)

cat("Printing every node's CPT (also exported to cpts_export.csv):\n\n")
cpt_rows <- list()
for (node in nodes(fitted_bn)) {
  cpt <- fitted_bn[[node]]$prob
  cat(strrep("=", 70), "\n")
  cat(sprintf("Node: %s\n", node))
  cat(strrep("=", 70), "\n")
  print(cpt)
  cat("\n")

  # Convert to a unified long format so nodes with different numbers of
  # parents can all be combined into one CSV: node, a single
  # "parent_config" string describing all parent values (or "(none)"
  # if the node has no parents), node_value, probability.
  df <- as.data.frame(as.table(cpt))
  # The node's own variable is always the first dimension; everything
  # else is a parent variable.
  value_col <- names(df)[1]
  parent_cols <- setdiff(names(df), c(value_col, "Freq"))

  if (length(parent_cols) == 0) {
    parent_config <- rep("(none)", nrow(df))
  } else {
    parent_config <- apply(df[parent_cols], 1, function(row) {
      paste(paste(parent_cols, row, sep = "="), collapse = "; ")
    })
  }

  unified <- data.frame(
    node = node,
    node_value = df[[value_col]],
    parent_config = parent_config,
    probability = df$Freq,
    stringsAsFactors = FALSE
  )
  cpt_rows[[node]] <- unified
}
all_cpts <- do.call(rbind, cpt_rows)
write.csv(all_cpts, "cpts_export.csv", row.names = FALSE)
cat("Saved cpts_export.csv\n")


# =======================================================================
# STEP 5: k-fold cross-validation with log-likelihood loss (paper Eq. 21,
# k = 10 to match the paper's default)
# =======================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("STEP 5: 10-fold cross-validation (log-likelihood loss)\n")
cat(strrep("=", 70), "\n")

cv_result <- tryCatch({
  bn.cv(
    data, bn = "mmhc", loss = "logl",
    k = 10,
    algorithm.args = list(blacklist = bl, maximize.args = list(score = "bde", iss = 10)),
    fit = "bayes",
    fit.args = list(iss = 10)
  )
}, error = function(e) {
  cat(sprintf("NOTE: 10-fold CV failed (%s).\n", conditionMessage(e)))
  cat("Retrying with k=5 (fewer folds means more data per fold, often more stable).\n")
  tryCatch({
    bn.cv(
      data, bn = "mmhc", loss = "logl",
      k = 5,
      algorithm.args = list(blacklist = bl, maximize.args = list(score = "bde", iss = 10)),
      fit = "bayes",
      fit.args = list(iss = 10)
    )
  }, error = function(e2) {
    cat(sprintf("NOTE: 5-fold CV also failed (%s). Skipping CV step.\n", conditionMessage(e2)))
    NULL
  })
})

if (!is.null(cv_result)) {
  losses <- sapply(cv_result, function(x) x$loss)
  cat(sprintf("Per-fold log-loss: %s\n", paste(round(losses, 4), collapse = ", ")))
  cat(sprintf("Mean log-loss: %.4f (std %.4f)\n", mean(losses), sd(losses)))
  cat("Lower log-loss = better fit (paper compares this against RSMAX2).\n")
  write.csv(data.frame(fold = seq_along(losses), logloss = losses),
            "kfold_logloss.csv", row.names = FALSE)
  cat("Saved kfold_logloss.csv\n")
}

# --- Comparison against RSMAX2, exactly as the paper does in Fig. 3 ---
cat("\n", strrep("=", 70), "\n", sep = "")
cat("Comparing MMHC vs RSMAX2 (paper's Fig. 3 comparison)\n")
cat(strrep("=", 70), "\n")

cv_result_rsmax2 <- tryCatch({
  bn.cv(
    data, bn = "rsmax2", loss = "logl",
    k = 10,
    algorithm.args = list(blacklist = bl, maximize.args = list(score = "bde", iss = 10)),
    fit = "bayes",
    fit.args = list(iss = 10)
  )
}, error = function(e) {
  cat(sprintf("NOTE: RSMAX2 CV failed (%s). Skipping comparison.\n", conditionMessage(e)))
  NULL
})

if (!is.null(cv_result_rsmax2) && !is.null(cv_result)) {
  losses_rsmax2 <- sapply(cv_result_rsmax2, function(x) x$loss)
  cat(sprintf("RSMAX2 per-fold log-loss: %s\n", paste(round(losses_rsmax2, 4), collapse = ", ")))
  cat(sprintf("RSMAX2 mean log-loss: %.4f (std %.4f)\n", mean(losses_rsmax2), sd(losses_rsmax2)))

  comparison <- data.frame(
    algorithm = c("MMHC", "RSMAX2"),
    mean_logloss = c(mean(losses), mean(losses_rsmax2)),
    sd_logloss = c(sd(losses), sd(losses_rsmax2))
  )
  cat("\nComparison summary:\n")
  print(comparison, row.names = FALSE)
  write.csv(comparison, "mmhc_vs_rsmax2.csv", row.names = FALSE)
  cat("Saved mmhc_vs_rsmax2.csv\n")

  if (mean(losses) < mean(losses_rsmax2)) {
    cat("\n=> MMHC has lower log-loss than RSMAX2 on your data ",
        "(matches the paper's own finding).\n", sep = "")
  } else {
    cat("\n=> RSMAX2 has lower log-loss than MMHC on your data ",
        "(differs from the paper's finding -- plausible given your ",
        "smaller/different dataset).\n", sep = "")
  }
}


# =======================================================================
# STEP 6: Mutual information between every causal factor and result event
# (Eq. 22 in the paper)
# =======================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("STEP 6: Mutual information ranking (causal factor -> result event)\n")
cat(strrep("=", 70), "\n")

mutual_information <- function(x, y) {
  tab <- table(x, y)
  n <- sum(tab)
  pxy <- tab / n
  px <- rowSums(pxy)
  py <- colSums(pxy)
  mi <- 0
  for (i in seq_len(nrow(pxy))) {
    for (j in seq_len(ncol(pxy))) {
      if (pxy[i, j] > 0) {
        mi <- mi + pxy[i, j] * log2(pxy[i, j] / (px[i] * py[j]))
      }
    }
  }
  return(mi)
}

mi_records <- data.frame(causal_factor = character(0), result_event = character(0), MI = numeric(0))
for (r in result_nodes) {
  for (c in causal_nodes) {
    mi <- mutual_information(data[[c]], data[[r]])
    mi_records <- rbind(mi_records, data.frame(causal_factor = c, result_event = r, MI = mi))
  }
}
mi_records <- mi_records[order(-mi_records$MI), ]

cat("\nTop 10 (causal_factor -> result_event) pairs by mutual information:\n")
print(head(mi_records, 10), row.names = FALSE)

agg <- aggregate(MI ~ causal_factor, data = mi_records, FUN = max)
agg <- agg[order(-agg$MI), ]
cat("\nTop 10 causal factors overall (paper's Table 4 analogue):\n")
print(head(agg, 10), row.names = FALSE)

write.csv(mi_records, "mutual_information_ranking.csv", row.names = FALSE)
cat("\nSaved mutual_information_ranking.csv\n")

# --- Table 1 style: prior probability of every causal factor node ---
cat("\n", strrep("=", 70), "\n", sep = "")
cat("Table 1 (paper style): Prior probability of each causal factor node\n")
cat(strrep("=", 70), "\n")

prior_probs <- sapply(causal_nodes, function(n) mean(as.numeric(as.character(data[[n]])) == 1))
prior_table <- data.frame(node = causal_nodes, prior_probability = round(prior_probs, 4))
prior_table <- prior_table[order(-prior_table$prior_probability), ]

cat("\nAll causal factor nodes, sorted by prior probability (highest first):\n")
print(prior_table, row.names = FALSE)

write.csv(prior_table, "prior_probabilities.csv", row.names = FALSE)
cat("\nSaved prior_probabilities.csv\n")

# --- Table 3 style: top-5 causal factors for each high-risk result event ---
# The paper names 4 specific high-risk result events (Section 4.3):
#   RE04 (Air Traffic Control - Separated Traffic)
#   RE05 (Aircraft - Aircraft Damaged)
#   RE14 (Flight Crew - Inflight Shutdown)
#   RE29 (General - Physical Injury / Incapacitation)
# For each, list its top-5 causal factors by mutual information, exactly
# as the paper's Table 3 does. If a given high-risk event isn't present
# in your dataset (e.g. dropped as an all-zero node), it's skipped with
# a note rather than causing an error.
cat("\n", strrep("=", 70), "\n", sep = "")
cat("Table 3 (paper style): Top-5 causal factors per high-risk result event\n")
cat(strrep("=", 70), "\n")

high_risk_events <- c("RE04", "RE05", "RE14", "RE29")
table3_rows <- list()

for (re in high_risk_events) {
  if (!(re %in% result_nodes)) {
    cat(sprintf("\n%s: not present in your dataset (likely dropped as an all-zero node). Skipping.\n", re))
    next
  }
  subset_mi <- mi_records[mi_records$result_event == re, ]
  subset_mi <- subset_mi[order(-subset_mi$MI), ]
  top5 <- head(subset_mi, 5)
  total_mi <- sum(subset_mi$MI)
  top5$percentage <- if (total_mi > 0) round(100 * top5$MI / total_mi, 2) else NA

  cat(sprintf("\nResult event: %s\n", re))
  print(top5[, c("causal_factor", "MI", "percentage")], row.names = FALSE)

  table3_rows[[re]] <- top5[, c("result_event", "causal_factor", "MI", "percentage")]
}

if (length(table3_rows) > 0) {
  table3_full <- do.call(rbind, table3_rows)
  write.csv(table3_full, "table3_high_risk_causal_factors.csv", row.names = FALSE)
  cat("\nSaved table3_high_risk_causal_factors.csv\n")
}


# =======================================================================
# STEP 7: Network diagram (igraph), analogous to the paper's Fig. 1
# =======================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("STEP 7: Rendering network diagram\n")
cat(strrep("=", 70), "\n")

g <- graph_from_data_frame(arcs_df, directed = TRUE, vertices = names(node_family))
deg <- degree(g, mode = "all")
V(g)$size <- 4 + 1.2 * deg
V(g)$color <- ifelse(node_family[V(g)$name] == "causal", "#D85A30", "#1D9E75")

isolated <- which(deg == 0)
if (length(isolated) > 0) {
  cat(sprintf("Removing %d isolated (unconnected) nodes.\n", length(isolated)))
  g <- delete_vertices(g, isolated)
}
cat(sprintf("Plotting %d nodes and %d edges.\n", vcount(g), ecount(g)))

png("catssr_bn_network.png", width = 2000, height = 1600, res = 150)
set.seed(42)
layout <- layout_with_fr(g)
plot(
  g, layout = layout,
  vertex.label.cex = 0.6, vertex.label.color = "black",
  edge.arrow.size = 0.25, edge.color = "#B4B2A9",
  main = sprintf("CATSSR-BN -- learned network structure (bnlearn MMHC)\n(%d nodes, %d edges)",
                  vcount(g), ecount(g))
)
legend("topleft",
       legend = c("Causal factor event (AE/HE/WE)", "Result event (RE)"),
       col = c("#D85A30", "#1D9E75"), pch = 19, pt.cex = 1.5, bty = "n")
dev.off()
cat("Saved catssr_bn_network.png\n")

cat("\nAll done. Outputs: bn_edges.csv, cpts_export.csv, ",
    "mutual_information_ranking.csv, catssr_bn_network.png\n", sep = "")