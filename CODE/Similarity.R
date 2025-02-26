# R script to run the similarity snow profile assessment
# loading library
library(dtw)
library(sarp.snowprofile)
library(sarp.snowprofile.alignment)
#library(survival)

 grainDict <- setColoursGrainType('IACS2')
#grainDict <- setColoursGrainType('SARP')

# --- AUX functions --- #
FCtoDH <- function(profile){
  a <- as.character(c(levels(profile$layers$gtype), 'DH'))
  levels(profile$layers$gtype) <- as.factor(a)

  #b <- as.character(c(levels(profile$layers$gtype_sec), 'DH'))
  #levels(profile$layers$gtype_sec) <- as.factor(b)

  for (idx in 1:nrow(profile$layers))
  {
    if (profile$layers[idx,]$gtype == 'FC')
    {
      if (profile$layers[idx,]$gsize >= 1.5)
      {
        profile$layers[idx,]$gtype <- "DH"
      }
    }
   # if (profile$layers[idx,]$gtype_sec == 'FC')
   #  {
   #   if (profile$layers[idx,]$gsize >= 1.5)
  #   {
   #     profile$layers[idx,]$gtype_sec <- "DH"
    #  }
    #}
  }
  return(profile)
}

IFtoMFcr<- function(profile){

  a <- as.character(c(levels(profile$layers$gtype), 'MFcr', 'MF'))
  levels(profile$layers$gtype) <- as.factor(a)

  # b <- as.character(c(levels(profile$layers$gtype_sec), 'MFcr', 'MF'))
  # levels(profile$layers$gtype_sec) <- as.factor(b)
  
  for (idx in seq_along(profile$layers$gtype)){
    if (profile$layers$gtype[idx]=='IF'){
      profile$layers$gtype[idx] = 'MFcr'
      # profile$layers$gtype_sec[idx] = 'MF'
      profile$layers$gsize[idx] = 0.5
    }
    if (profile$layers$gtype[idx]=='MFcr'){
      profile$layers$hardness[idx] = 5.0
    }
  }
  return(profile)
}

hardnessN_to_handHardness <- function(profile){
  hardness_dict <- c(
  "5"   = 1,
  "19"  = 1.5,
  "26"  = 1.5,
  "28"  = 1.5,
  "39"  = 1.75,
  "46"  = 1.75,
  "51"  = 2,
  "52"  = 2,
  "59"  = 2,
  "62"  = 2,
  "102" = 2.5,
  "174" = 3,
  "270" = 3.5,
  "390" = 3.75,
  "538" = 4,
  "713" = 4.5,
  "918" = 5
  )
  for (idx in seq_along(profile$layers$gtype)){
    if (profile$type == "modeled"){
      profile$layers$hardness[idx] = hardness_dict[as.character(round(profile$layers$hardness[idx]))]
    }
  }
  return(profile)
}

profsimilarity <- function(caaml_file){
PROF_OBS    <- paste(caaml_folder, caaml_file, sep="/") # paste(…, sep="", collapse=NULL)
man         <- snowprofileCaaml(PROF_OBS, sourceType = "manual")

man         <- FCtoDH(man)
man         <- IFtoMFcr(man)
man         <- computeRTA(man)
man         <- computeTSA(man)


mod_dates   <- scanProfileDates(PROFS_MOD)
i_profDate  <- which(abs(mod_dates - man$datetime) == min(abs(mod_dates - man$datetime)))
prof_date   <- mod_dates[i_profDate]
mod         <- snowprofilePro(PROFS_MOD, ProfileDate=prof_date, tz='UTC')
# - Select for station simulations - #
mod         <- hardnessN_to_handHardness(mod)
# - Select for station simulations - #
mod         <- computeRTA(mod)
mod         <- computeTSA(mod)
mod         <- IFtoMFcr(mod)

# - Alignment - #
# alignment_type = "HerlaEtAl2021"
alignment_type = "rta_wldetection"
#alignment_type = "tsa_wldetection"
#alignment_type = "wsum_scaled"
#alignment_type = "rta_scaling"
#alignment_type = "layerwise"

my_dtwAlignment <- function(query, ref, alignment_type){
    dtwAlignment<-  dtwSP(query, ref,
                            # prefLayerWeights = layerWeightingMat(FALSE),
                            dims = c("gtype", "hardness"), weights = c(gType=0.7,hardness=0.3),
                            resamplingRate = 0.5,
                            rescale2refHS = TRUE,
                            # windowFunction = warpWindowSP,
                            window.size = 0.6,
                            step.pattern = dtw::symmetricP1,
                            # step.pattern = symmetric2,
                            open.end = TRUE, checkGlobalAlignment = TRUE,
                            keep.internals = TRUE,
                            simType = alignment_type)
    return(dtwAlignment)
}

# interactiveAlignment(query = mod, ref = man)
alignment <- my_dtwAlignment(mod,man, alignment_type=alignment_type)
simdf <- simSP(alignment$reference, alignment$queryWarped, simType=alignment_type, nonMatchedSim = 0.5, nonMatchedThickness=10, verbose=TRUE, returnDF = TRUE)

return(simdf)
}


# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Print received arguments
#print(paste("Argument 1:", args[1]))
#print(paste("Argument 2:", args[2]))

# --- AXLIZ --- #
#pro_folder   <- './FID'
ROOT <- args[1]
pro_folder   <- args[2]
path_topro_folder <- paste(ROOT, pro_folder, sep = '')
#pro_file    <- 'axliz_prof1.pro'

simdf <- data.frame(simu_id = c(), simi = c(), wl = c(), cr = c(), pp = c(), bulk = c())
simdfall <- data.frame(simu_id = c(),date = c(), simi = c(), wl = c(), cr = c(), pp = c(), bulk = c())

pro_file     <- 'output/30_55_FID_1.0_PSUM_15m.pro'
PROFS_MOD   <- paste(path_topro_folder, pro_file, sep="/")
caaml_folder <- './profil_2018-2019'
#caaml_file   <- '20190208.caaml'
# --- AXLIZ --- #

#profile_vector <- c('20181104.caaml','20181210.caaml','20190108.caaml','20190208.caaml','20190308.caaml','20190405.caaml')
profile_vector <- c('20190108.caaml','20190208.caaml','20190308.caaml')
simi = c()
simi_wl = c()
simi_cr = c()
simi_pp = c()
simi_bulk = c()
for (i in profile_vector){
    print(i)
    simval <- profsimilarity(i)
    simi <- append(simi,simval$sim)
    print(simval$sim)
    simi_wl <- append(simi_wl,simval$simDF$wl)
    simi_cr <- append(simi_cr,simval$simDF$cr)
    simi_pp <- append(simi_pp,simval$simDF$pp)
    simi_bulk <- append(simi_bulk,simval$simDF$bulk)
    simdate <- data.frame( sim_id = c(pro_folder), date = c(substring(i,1,8)), simi = c(simval$sim), wl = c(simval$simDF$wl), cr = c(simval$simDF$cr), pp = c(simval$simDF$pp), bulk = c(simval$simDF$bulk))
    simdfall <- rbind(simdfall,simdate)}
simdf1 <- data.frame( sim_id = c(pro_folder), simi = c(mean(simi)), wl = c(mean(simi_wl)), cr = c(mean(simi_cr)), pp = c(mean(simi_pp)), bulk = c(mean(simi_bulk)))
#print(paste('simi:', 1 - mean(simi)))
write.csv(simdf1, paste('./FID_OPTIMIZATION/profil', paste(pro_folder,"sim_metstation.csv", sep = '_'), sep = '/'), row.names = FALSE)
