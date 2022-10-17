library(scImpute)
scimpute(
      data = ("m.csv"), 
      infile = "csv",          
      outfile = "csv",          
      out_dir = "Zimpute",
      labeled = FALSE,          
      drop_thre = 0.6,          
      Kcluster = 13,           
      ncores = 1) 


