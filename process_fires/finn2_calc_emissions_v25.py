#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINNv2 Emissions Calculation - Version 2.5
Calculate emissions from pre-processed fire locations
1) reads fuel loads for each vegetation type for each global region
2) reads emission factors file for aerosols and a few gases for each vegetation type
3) reads processed fires file (from FINN preprocessor)
4) for each fire read matches LCT to generic vegetation, determines biomass burned,
   calculates emissions
5) from total NMVOC calcuated in step 4 calculates emissions for specific VOCS
   for MOZART, SAPRC99 and GEOS-Chem chemical mechanisms
6) writes output to text files
"""

import numpy as np
import pandas as pd
import os
import time
from datetime import datetime, timedelta

def x_finn2_calc_emissions_v2_5(file_in, simid, year_emis, date_lab, todaydate, path_out):
    
    print('Started on: ', time.ctime())
    t0 = time.time()  # Procedure start time in seconds
    
    finnver = 'v2.5'
    
    # Close all open files
    # (Not needed in Python as file handles are closed automatically)
    
    sdate_emis = date_lab
    
    # Open a log file
    logfile = path_out + 'LOG_calcemis_FINN'+finnver+'_'+simid+'_'+sdate_emis+'_'+todaydate+'.txt'
    ilun_log = open(logfile, 'w')
    print('writing log file: ', logfile)
    
    log_genveg = path_out + 'LOG_genveg_calcemis_FINN'+finnver+'_'+simid+'_'+sdate_emis+'_'+todaydate+'.txt'
    print('log of genveg assignment: ', log_genveg)
    ilun_gv = open(log_genveg, 'w')
    ilun_gv.write('i, jday, fireid, polyid, lat, lon, lct_in, lct, tree_in, tree, herb_in, herb, bare_in, bare, flct, genveg\n')
    
    #**************
    # Standard input files (Fuel loads, EFs)
    file_fuelloads = '/data14a/FINN/finnv2.3/FINN2_inputs/Fuel_LOADS_NEW_022019.csv' 
    file_usfuel = '/data14a/FINN/finnv2.3/FINN2_inputs/LCTFuelLoad_fuel4_revisit20190521.csv'
    file_efs = '/data14a/FINN/finnv2.4/EFs_byGenVeg_c20210601.csv'
    file_VOCsplit_M = '/data14a/FINN/finnv2.4/NMOCfrac_byGenVeg_MOZ.csv'
    file_VOCsplit_S = '/data14a/FINN/finnv2.4/NMOCfrac_byGenVeg_SAPRC.csv'
    file_VOCsplit_G = '/data14a/FINN/finnv2.4/NMOCfrac_byGenVeg_GEOSCHEM.csv'
    
    #  READ IN FUEL LOADING FILE: fuel loads for 5 veg types, for 13 regions
    #  ALL FUEL INPUTS ARE IN g/m2 [-1 for missing values]
    fuel = pd.read_csv(file_fuelloads)
    fueltypes = fuel.columns.tolist()
    print(file_fuelloads, ' Contains: ', fueltypes)
    ilun_log.write(f"{file_fuelloads} Contains: {fueltypes}\n")
    # print('expected: region#, trop.for., temp.for., bor.for., woodySav, grassSav')
    if (fueltypes[0] != 'GlobalRegion') or (fueltypes[5] != 'SavannaGrasslands'):
        raise ValueError('wrong format')
    
    #   Set up fuel arrays
    regfuel = fuel['GlobalRegion'].values
    tffuel = fuel['TropicalForest'].values
    tefuel = fuel['TemperateForest'].values
    bffuel = fuel['BorealForest'].values
    wsfuel = fuel['WoodySavanna'].values
    grfuel = fuel['SavannaGrasslands'].values
    
    # READ in a secondary fuel loading file for use in US ONLY 
    #   has fuel loads for tree and herb for each LCT 
    LCTfuel = pd.read_csv(file_usfuel)
    usfueltypes = LCTfuel.columns.tolist()
    print(file_usfuel, ' Contains: ', usfueltypes)
    ilun_log.write(f"{file_usfuel} Contains: {usfueltypes}\n")
    # print('expected: Code,TREE,HERB')
    if usfueltypes[2] != 'HERB':
        raise ValueError('wrong format')
    
    lctfuelid = LCTfuel['Code'].values
    lcttree = LCTfuel['TREE'].values
    lctherb = LCTfuel['HERB'].values
    
    # READ IN EMISSION FACTOR FILE [g compound emitted per kg dry biomass burned]
    #   EFs for arbitrary number of species for GenVeg 1-6,9
    print('Reading ', file_efs)
    ilun_log.write(f"Reading {file_efs}\n")
    ntype = 7  # emission factors for genveg=1-6,9
    
    with open(file_efs, 'r') as ilun:
        sdum = ilun.readline().strip()  # header/title
        sdum = ilun.readline().strip()  # column names
        colnames_ef = sdum.split(',')
        ilun_log.write(f"{file_efs} contains: {colnames_ef}\n")
        # print(' expected: GenVegType,GenVegDescript, {species ...}')
        ncols = len(colnames_ef)
        ef_genveg = np.zeros(ntype, dtype=int)  # genveg index for emission factors
        # assumes first 2 cols are genveg, GenVegDesc, remaining columns are species EFs
        nspec = ncols - 2
        ef_species = colnames_ef[2:ncols]
        # assumes 3rd row has molecular weights
        sdum = ilun.readline().strip()  # MWs
        parts = sdum.split(',')
        mws = np.array([float(x) for x in parts[2:ncols]])
        # read emission factors for all species
        emisfac = np.zeros((ntype, nspec), dtype=float)
        for itype in range(ntype):
            sdum = ilun.readline().strip()
            cols = sdum.split(',')
            ef_genveg[itype] = int(cols[0])
            emisfac[itype, :] = [float(x) for x in cols[2:ncols]]  # emission factors in g/kg
    
    print('Have EFs for: ', ef_species)
    print(' for GenVeg Types: ', ef_genveg)
    for ispec in range(nspec):
        ilun_log.write(f"{ef_species[ispec]:<12}{int(mws[ispec]):4}" + 
                       "".join([f"{emisfac[i, ispec]:12.3f}" for i in range(7)]) + "\n")
    
    #***************************************************************************************
    # READ FIRE AND LAND COVER INPUT FILE (CREATED WITH PREPROCESSOR)
    # **************************************************************************************
    #  determine genveg
    #  determine fuel loads and biomass burned
    #  calculate emissions  
    #--------------
    # Read first line of fire file
    # Set up arrays for saving emissions and other info
    # -------------
    with open(file_in, 'r') as f:
        nfires = sum(1 for _ in f) - 1  # subtract header line
    
    print('# fires: ', nfires)
    with open(file_in, 'r') as ilun_in:
        sdum = ilun_in.readline().strip()
        colnames_fires = sdum.split(',')
        print(file_in, ' contains: ', colnames_fires)
        ilun_log.write(f"{file_in} contains: {colnames_fires}\n")
        
        #polyid,fireid,cen_lon,cen_lat,acq_date_lst,area_sqkm,v_lct,f_lct,v_tree,v_herb,v_bare,v_regnum
        #1,4322940,151.467734804981,-33.1993797560266,2018-12-30,1.71356443131792,8,0.859375,36.078125,58.515625,5.984375,12
        # polyid fireid cen_lon cen_lat acq_date_lst area_sqkm v_lct f_lct v_tree v_herb v_bare v_regnum fireid0
        
        #----------------
        #  Read each line of input fire file 
        #  Determine vegetation type, area, biomass burned
        #  Save to arrays only valid fire points (correct date, valid vegetation, etc)
        #----------------
        em_jday = np.zeros(nfires, dtype=int)
        em_date = np.zeros(nfires, dtype=int)
        em_polyid = np.zeros(nfires, dtype=int)
        em_fireid = np.zeros(nfires, dtype=int)
        em_lat = np.zeros(nfires, dtype=float)
        em_lon = np.zeros(nfires, dtype=float)
        em_area = np.zeros(nfires, dtype=float)
        em_bmass = np.zeros(nfires, dtype=float)
        em_genveg = np.zeros(nfires, dtype=int)
        
        igood = 0
        iskip_yr = 0
        iskip_reg = 0
        
        for i in range(nfires):
            sdum = ilun_in.readline().strip()
            parts = sdum.split(',')
            if len(parts) != 12:
                ilun_log.write(f"input line wrong size: {len(parts)} {sdum}\n")
                continue  # skipfire
            
            dateparts = parts[4].split('-')
            yy = int(dateparts[0])
            mm = int(dateparts[1])
            dd = int(dateparts[2])
            if yy != year_emis:
                # ilun_log.write('wrong year ' + str(yy) + '\n')
                iskip_yr += 1
                continue  # skipfire
            
            date = int(''.join(dateparts))
            jday = (datetime(yy, mm, dd) - datetime(yy, 1, 1)).days + 1
            polyid = int(parts[0])
            fireid = int(parts[1])
            lon = float(parts[2])
            lat = float(parts[3])
            area = float(parts[5])
            lct = int(parts[6])
            flct = float(parts[7])
            tree = float(parts[8])
            herb = float(parts[9])
            bare = float(parts[10])
            globreg = int(parts[11])
            
            # remove values of -9999 from VCF inputs
            if tree < 0.:
                tree = 0.
            if herb < 0.:
                herb = 0.
            if bare < 0.:
                bare = 0.
            
            # Calculate the total cover from the VCF product (CHECK TO MAKE SURE PERCENTAGES ADD TO 100%)
            totcov = tree + herb + bare
            
            # Remove fires with no LCT assignment or in water bodies or snow/ice assigned by LCT
            # LCT:
            # 12, 14: cropland
            # 13: urban
            # 15: permanent snow or ice
            # 16: barren
            # 17: water
            # 255: unclassified
            if (lct >= 17) or (lct <= 0) or (lct == 15):
                ilun_log.write(f"Fire {i} removed: lct = {lct}\n")
                continue  # skipfire
            
            if (totcov >= 240.) or (totcov < 1.):
                ilun_log.write(f"Fire {i} removed. totcov={int(totcov)}\n")
                continue  # skipfire
            
            lct_in = lct
            tree_in = tree
            herb_in = herb
            bare_in = bare
            
            # Scale VCF product to sum to 100.
            if (totcov > 101.) or (totcov < 99.):
                totcov_orig = totcov
                tree = tree_in * 100. / totcov
                herb = herb_in * 100. / totcov
                bare = bare_in * 100. / totcov
                totcov = bare + herb + tree
                ilun_log.write(f"Fire {i} had totcov adjusted from: {int(totcov_orig)} to: {int(totcov)}\n")
            
            # Fires with 100% bare cover reassign cover values based on LCT assignment
            if bare >= 99.9:
                if lct <= 5:  # Assign forest to the pixel
                    tree = 60.
                    herb = 40.
                    bare = 0.
                if (lct >= 6 and lct <= 8) or (lct == 11) or (lct == 14):  # Assign woody savanna to the pixel
                    tree = 50.
                    herb = 50.
                    bare = 0.
                if (lct == 9) or (lct == 10) or (lct == 12) or (lct == 13) or (lct == 16):  # Assign grassland to the pixel
                    tree = 20.
                    herb = 80.
                    bare = 0.
                ilun_log.write(f"Fire {i} with LCT={lct} had 100% bare adjusted to T/H/B: {int(tree)}/{int(herb)}/{int(bare)}\n")
            
            # ######################################################
            # Assign Generic land cover to fire based on
            #   global location and lct information
            # ######################################################
            #Generic land cover codes (genveg) are as follows:
            #1 grassland
            #2 shrub
            #3 Tropical Forest
            #4 Temperate Forest
            #5 Boreal Forest
            #6 Temperate Evergreen Forest
            #7 Pasture
            #8 Rice
            #9 Crop (generic)
            #10  Wheat
            #11  Cotton
            #12  Soy
            #13  Corn
            #14  Sorghum
            #15  Sugar Cane
            genveg = -999
            
            if lct == 1:  # Evergreen Needleleaf Forest to Boreal or Temperate Evergreen
                if lat > 50.:
                    genveg = 5
                else:
                    genveg = 6
            elif lct == 2:
                if lat >= -23.5 and lat <= 23.5:
                    genveg = 3  # Tropical Forest
                else:
                    genveg = 4  # Temperate Forest
            elif lct == 3:  # deciduous Needleleaf Forest to Boreal or Temperate forest
                if lat > 50.:
                    genveg = 5
                else:
                    genveg = 4
            elif lct == 4:
                genveg = 4  # Temperate Forest
            elif lct == 5:  # Mixed Forest, assign type by latitude
                if lat > 50.:
                    genveg = 5
                elif lat >= -23.5 and lat <= 23.5:
                    genveg = 3
                else:
                    genveg = 4
            elif lct == 6 or lct == 7 or lct == 8:
                genveg = 2  # Woody Savanna or Shrubs
            elif lct == 9 or lct == 10 or lct == 11:
                genveg = 1  # Grasslands and Savanna
            elif lct == 12:
                genveg = 9  # Croplands
            elif lct == 13:  # Urban
                if tree < 40.:
                    genveg = 1  # grasslands
                    lct = 10  # set to grassland
                elif tree >= 40. and tree < 60.:
                    genveg = 2  # woody savannas
                    lct = 8  # set to woody savanna
                elif tree >= 60.:  # assign forest based on latitude
                    if lat > 50.:
                        genveg = 5
                        lct = 1  # set to evergreen needleleaf forest
                    else:
                        if lat >= -30. and lat <= 30.:
                            genveg = 3
                        else:
                            genveg = 4
                        lct = 5  # set to mixed forest
            elif lct == 14 or lct == 16:
                genveg = 1
            else:
                genveg = -1
            
            if genveg <= 0:
                ilun_log.write(f"Fire {i} does not have genveg set. lat,lon, LCT = {lat},{lon},{lct}\n")
                continue  # skipfire
            
            ilun_gv.write(f"{i:10d}{jday:5d}{fireid:10d}{polyid:10d}{lat:9.3f}{lon:9.3f}{lct_in:3d}{lct:3d}" +
                           f"{tree_in:6.1f}{tree:6.1f}{herb_in:6.1f}{herb:6.1f}{bare_in:6.1f}{bare:6.1f}{flct:6.1f}{genveg:3d}\n")
            
            # ####################################################
            # Assign Fuel Loads based on Generic land cover
            #   and global region location
            #   units are in g dry mass/m2
            # ####################################################
            reg = globreg - 1  # locate global region, get index
            if reg < 0 or reg > 100:
                ilun_log.write(f"Fire {i} removed. global region: {globreg} lon, lat: {lon:.1f}, {lat:.1f}\n")
                continue  # skipfire
            
            # Assign biomass density according to veg. type
            bmass1 = -1.
            if genveg == 1:
                bmass1 = grfuel[reg]
            elif genveg == 2:
                bmass1 = wsfuel[reg]
            elif genveg == 3:
                bmass1 = tffuel[reg]
            elif genveg == 4:
                bmass1 = tefuel[reg]
            elif genveg == 5:
                bmass1 = bffuel[reg]
            elif genveg == 6:
                bmass1 = tefuel[reg]
            elif genveg == 9:
                bmass1 = 902.
            
            # Assign boreal forests in Southern Asia the biomass density of the temperate forest for the region
            if genveg == 5 and globreg == 11:
                bmass1 = tefuel[reg]
            
            if bmass1 < 0.:
                ilun_log.write(f"BMASS1 < 0: Fire {i} removed. genveg = {genveg} globreg = {globreg} reg = {reg}\n")
                continue  # skipfire
            
            # *****************************************************************************************
            # Assign Burning Efficiencies based on Generic land cover (Hoezelmann et al. [2004] Table 5
            # *****************************************************************************************
            # ASSIGN CF VALUES (Combustion Factors)
            if tree > 60.:  # FOREST
                # Values from Table 3 Ito and Penner [2004]
                CF1 = 0.30          # Live Woody
                CF3 = 0.90          # Leafy Biomass
                # CF4 = 0.90        # Herbaceous Biomass
                # CF5 = 0.90        # Litter Biomass
                # CF6 = 0.30        # Dead woody
            if tree > 40. and tree <= 60.:  # WOODLAND
                CF3 = np.exp(-0.013*tree)  # Apply to all herbaceous fuels
                CF1 = 0.30                # Apply to all coarse fuels in woodlands
                # From Ito and Penner [2004]
            if tree <= 40.:  # GRASSLAND
                CF3 = 0.98  # Range is between 0.44 and 0.98 - Assumed UPPER LIMIT!
            
            # *******************************************************************************************
            # Calculate the biomass burned of each classification (herbaceous, woody, and forest)
            # These are in units of g dry matter/m2
            # Bmass is the total burned biomass
            # herbbm is the Herbaceous biomass burned
            # coarsebm is the Woody biomass burned
            
            coarsebm = bmass1
            herbbm = grfuel[reg]
            
            # Determine if in North America and use updated fuel loading for North America (Global Region 1)
            if globreg == 1:
                # Assign coarse and herb biomass based on lct
                coarsebm = lcttree[lct]
                herbbm = lctherb[lct]
            
            # Grasslands
            if tree <= 40.:
                Bmass = ((herb/100.)*herbbm*CF3) + ((tree/100.)*herbbm*CF3)
                # Assumed here that litter biomass = herbaceous biomass and that the percent tree
                # in a grassland cell contributes to fire fuels
                # Assuming here that the duff and litter around trees burn
            # Woodlands
            if tree > 40. and tree <= 60.:
                Bmass = ((herb/100.)*herbbm*CF3) + ((tree/100.)*(herbbm*CF3+coarsebm*CF1))
            # Forests
            if tree > 60.:
                Bmass = ((herb/100.)*herbbm*CF3) + ((tree/100.)*(herbbm*CF3+coarsebm*CF1))
            
            # Convert units to be consistent; adjust area burned for vegetation and bare fraction
            bmass = Bmass / 1000.  # convert g-dm/m2 to kg-dm/m2
            areanow = area * flct * 1.0e6  # convert km2 to m2
            area_bare = areanow * (bare / 100.)
            areanow = areanow - area_bare  # remove bare area from being burned
            if areanow < 1.:
                ilun_log.write(f"area = 0. area,flct,bare: {areanow},{flct},{bare}\n")
                continue  # skipfire
            
            em_jday[igood] = jday
            em_date[igood] = date
            em_polyid[igood] = polyid
            em_fireid[igood] = fireid
            em_lat[igood] = lat
            em_lon[igood] = lon
            em_area[igood] = areanow
            em_bmass[igood] = bmass
            em_genveg[igood] = genveg
            
            igood += 1
    
    print('finished reading fire file')
    
    ngood = igood
    em_jday = em_jday[:ngood]
    em_date = em_date[:ngood]
    em_polyid = em_polyid[:ngood]
    em_fireid = em_fireid[:ngood]
    em_lat = em_lat[:ngood]
    em_lon = em_lon[:ngood]
    em_area = em_area[:ngood]
    em_bmass = em_bmass[:ngood]
    em_genveg = em_genveg[:ngood]
    
    # Sort fires by day
    indsort = np.argsort(em_jday)
    em_jday = em_jday[indsort]
    em_date = em_date[indsort]
    em_polyid = em_polyid[indsort]
    em_fireid = em_fireid[indsort]
    em_lat = em_lat[indsort]
    em_lon = em_lon[indsort]
    em_area = em_area[indsort]
    em_bmass = em_bmass[indsort]
    em_genveg = em_genveg[indsort]
    
    print('finished sorting arrays of good points', ngood)
    print(' # fires with emissions: ', ngood)
    print(' % of total fires saved: ', float(ngood)/float(nfires)*100.)
    
    ilun_log.write(f"# fires skipped because wrong year: {iskip_yr}\n")
    ilun_log.write(f"# fires skipped because no region assigned: {iskip_reg}\n")
    ilun_log.write(f"# fires with emissions: {ngood}\n")
    ilun_log.write(f"% of total fires saved: {float(ngood)/float(nfires)*100.}\n")
    
    # Read factors to convert NMOC [kg/day] to each VOC [moles/day]
    #  for MOZART, SAPRC and GEOS-Chem separately
    print('------ MOZART ------')
    ind_NMOC = np.where(np.array(ef_species) == 'NMOC')[0]
    ind_NMOC = ind_NMOC[0] if len(ind_NMOC) > 0 else -1
    have_file = os.path.exists(file_VOCsplit_M)
    
    if ind_NMOC >= 0 and have_file:
        print('index of NMOC: ', ind_NMOC, ' ', ef_species[ind_NMOC])
        print('Reading ', file_VOCsplit_M)
        
        with open(file_VOCsplit_M, 'r') as ilun_voc:
            sdum = ilun_voc.readline().strip()  # header line
            sdum = ilun_voc.readline().strip()  # column labels
            # GenVegIndex, EF_fuel_type, APIN,...
            colnames = sdum.split(',')
            nvocs_split = len(colnames) - 2
            vocnames = colnames[2:nvocs_split+2]
            print('MOZART VOC speciation for: ')
            print(vocnames)
            ntypes = 7
            voc_fraction = np.zeros((nvocs_split, ntypes))
            genveg_voc = np.zeros(ntypes, dtype=int)
            
            for itype in range(ntypes):
                sdum = ilun_voc.readline().strip()
                cols = sdum.split(',')
                genveg_voc[itype] = int(cols[0])
                for ivoc in range(nvocs_split):
                    voc_fraction[ivoc, itype] = float(cols[ivoc+2])
            # for ivoc in range(nvocs_split):
            #    print(vocnames[ivoc], voc_fraction[ivoc, :])
    else:
        print('Not calculating VOC speciation: NMOC not in Emission Factors file or cannot open ', file_VOCsplit_M)
        nvocs_split = 0
    
    # SET UP OUTPUT TEXT FILE for base and MOZART species
    outfile_txt = path_out + 'FINN'+finnver+'_'+simid + '_MOZART_'+sdate_emis+'_c'+todaydate+'.txt'
    print('Writing output to: ', outfile_txt)
    with open(outfile_txt, 'w') as ilun_out:
        species_list = ','.join(ef_species)
        vocs_list = ', ' + ','.join(vocnames) if nvocs_split > 0 else ''
        
        ilun_out.write('DAY,POLYID,FIREID,GENVEG,LATI,LONGI,AREA,BMASS,' + species_list + vocs_list + '\n')
        
        print('Calculating emissions... ')
        
        # ####################################################
        # CALCULATE EMISSIONS = area*BMASS*EF
        # ####################################################
        # Units: EF[g-species/kg-dm]*[kg/g] * Area[m2] * Bmass[kg-dm/m2]
        # Convert gas-phase species to [moles/fire/day] by scaling with molecular weight [kg/mole]
        # Keep aerosols in [kg/fire/day] (MW=1 from EFs file)
        
        for ifire in range(ngood):
            emis_spec = np.zeros(nspec)
            emis_vocs = np.zeros(nvocs_split) if nvocs_split > 0 else np.array([])
            
            itype = np.where(ef_genveg == em_genveg[ifire])[0]
            if len(itype) == 0:
                ilun_log.write(f"no EF for this genveg: {em_genveg[ifire]}\n")
                continue  # skipfire2
            itype = itype[0]
            
            for ispec in range(nspec):
                if mws[ispec] != 1.:
                    emis_spec[ispec] = emisfac[itype, ispec] * 1.e-3 * em_area[ifire] * em_bmass[ifire] / (mws[ispec] * 1.e-3)
                else:
                    emis_spec[ispec] = emisfac[itype, ispec] * 1.e-3 * em_area[ifire] * em_bmass[ifire]
            
            # Calculate emissions for VOCs (moles) as fraction of NMOC (kg-species)
            if nvocs_split > 0:
                igen = np.where(genveg_voc == em_genveg[ifire])[0]
                if len(igen) > 0:
                    for ivoc in range(nvocs_split):
                        emis_vocs[ivoc] = voc_fraction[ivoc, igen[0]] * emis_spec[ind_NMOC]
                else:
                    emis_vocs[:] = 0.
            
            # Write each fire to text file
            output_line = f"{em_jday[ifire]},{em_fireid[ifire]},{em_polyid[ifire]},"
            output_line += f"{em_genveg[ifire]},{em_lat[ifire]:.3f},{em_lon[ifire]:.3f},"
            output_line += f"{em_area[ifire]:.3e},{em_bmass[ifire]:.3e},"
            output_line += ','.join([f"{x:.3e}" for x in emis_spec])
            
            if nvocs_split > 0:
                output_line += ',' + ','.join([f"{x:.3e}" for x in emis_vocs])
            
            ilun_out.write(output_line + '\n')
    
    #---------------------------
    # CALCULATE SAPRC speciation
    # Read factors to convert NMOC [kg/day] to each VOC [moles/day]
    print('------ SAPRC ------')
    ind_NMOC = np.where(np.array(ef_species) == 'NMOC')[0]
    ind_NMOC = ind_NMOC[0] if len(ind_NMOC) > 0 else -1
    have_file = os.path.exists(file_VOCsplit_S)
    
    if ind_NMOC >= 0 and have_file:
        print('index of NMOC: ', ind_NMOC, ' ', ef_species[ind_NMOC])
        print('Reading ', file_VOCsplit_S)
        
        with open(file_VOCsplit_S, 'r') as ilun_voc:
            sdum = ilun_voc.readline().strip()  # header line
            sdum = ilun_voc.readline().strip()  # column labels
            # GenVegIndex, EF_fuel_type, species...
            colnames = sdum.split(',')
            nvocs_split = len(colnames) - 2
            vocnames = colnames[2:nvocs_split+2]
            print('SAPRC VOC speciation for: ')
            print(vocnames)
            ntypes = 7
            voc_fraction = np.zeros((nvocs_split, ntypes))
            genveg_voc = np.zeros(ntypes, dtype=int)
            
            for itype in range(ntypes):
                sdum = ilun_voc.readline().strip()
                cols = sdum.split(',')
                genveg_voc[itype] = int(cols[0])
                for ivoc in range(nvocs_split):
                    voc_fraction[ivoc, itype] = float(cols[ivoc+2])
            # for ivoc in range(nvocs_split):
            #    print(vocnames[ivoc], voc_fraction[ivoc, :])
    else:
        print('Not calculating VOC speciation: NMOC not in Emission Factors file or cannot open ', file_VOCsplit_S)
        nvocs_split = 0
    
    print('SAPRC #VOCs: ', nvocs_split)
    
    # SET UP OUTPUT TEXT FILE for base and SAPRC species
    outfile_txt = path_out + 'FINN'+finnver+'_'+simid + '_SAPRC_'+sdate_emis+'_c'+todaydate+'.txt'
    print('Writing output to: ', outfile_txt)
    with open(outfile_txt, 'w') as ilun_out:
        species_list = ','.join(ef_species)
        vocs_list = ', ' + ','.join(vocnames) if nvocs_split > 0 else ''
        
        ilun_out.write('DAY,POLYID,FIREID,GENVEG,LATI,LONGI,AREA,BMASS,' + species_list + vocs_list + '\n')
        
        print('Calculating emissions... ')
        
        for ifire in range(ngood):
            emis_spec = np.zeros(nspec)
            emis_vocs = np.zeros(nvocs_split) if nvocs_split > 0 else np.array([])
            
            itype = np.where(ef_genveg == em_genveg[ifire])[0]
            if len(itype) == 0:
                ilun_log.write(f"no EF for this genveg: {em_genveg[ifire]}\n")
                continue  # skipfire3
            itype = itype[0]
            
            for ispec in range(nspec):
                if mws[ispec] != 1.:
                    emis_spec[ispec] = emisfac[itype, ispec] * 1.e-3 * em_area[ifire] * em_bmass[ifire] / (mws[ispec] * 1.e-3)
                else:
                    emis_spec[ispec] = emisfac[itype, ispec] * 1.e-3 * em_area[ifire] * em_bmass[ifire]
            
            # Calculate emissions for VOCs (moles) as fraction of NMOC (kg-species)
            if nvocs_split > 0:
                igen = np.where(genveg_voc == em_genveg[ifire])[0]
                if len(igen) > 0:
                    for ivoc in range(nvocs_split):
                        emis_vocs[ivoc] = voc_fraction[ivoc, igen[0]] * emis_spec[ind_NMOC]
                else:
                    emis_vocs[:] = 0.
            
            # Write each fire to text file
            output_line = f"{em_jday[ifire]},{em_fireid[ifire]},{em_polyid[ifire]},"
            output_line += f"{em_genveg[ifire]},{em_lat[ifire]:.3f},{em_lon[ifire]:.3f},"
            output_line += f"{em_area[ifire]:.3e},{em_bmass[ifire]:.3e},"
            output_line += ','.join([f"{x:.3e}" for x in emis_spec])
            
            if nvocs_split > 0:
                output_line += ',' + ','.join([f"{x:.3e}" for x in emis_vocs])
            
            ilun_out.write(output_line + '\n')
    
    #---------------------------
    # CALCULATE GEOSCHEM speciation
    # Read factors to convert NMOC [kg/day] to each VOC [moles/day]
    print('------ GEOSCHEM ------')
    ind_NMOC = np.where(np.array(ef_species) == 'NMOC')[0]
    ind_NMOC = ind_NMOC[0] if len(ind_NMOC) > 0 else -1
    have_file = os.path.exists(file_VOCsplit_G)
    
    if ind_NMOC >= 0 and have_file:
        print('index of NMOC: ', ind_NMOC, ' ', ef_species[ind_NMOC])
        print('Reading ', file_VOCsplit_G)
        
        with open(file_VOCsplit_G, 'r') as ilun_voc:
            sdum = ilun_voc.readline().strip()  # header line
            sdum = ilun_voc.readline().strip()  # column labels
            # GenVegIndex, EF_fuel_type, species...
            colnames = sdum.split(',')
            nvocs_split = len(colnames) - 2
            vocnames = colnames[2:nvocs_split+2]
            print('GEOS-Chem VOC speciation for: ')
            print(vocnames)
            ntypes = 7
            voc_fraction = np.zeros((nvocs_split, ntypes))
            genveg_voc = np.zeros(ntypes, dtype=int)
            
            for itype in range(ntypes):
                sdum = ilun_voc.readline().strip()
                cols = sdum.split(',')
                genveg_voc[itype] = int(cols[0])
                for ivoc in range(nvocs_split):
                    voc_fraction[ivoc, itype] = float(cols[ivoc+2])
            # for ivoc in range(nvocs_split):
            #    print(vocnames[ivoc], voc_fraction[ivoc, :])
    else:
        print('Not calculating VOC speciation: NMOC not in Emission Factors file or cannot open ', file_VOCsplit_G)
        nvocs_split = 0
    
    print('GC #VOCS:', nvocs_split)
    
    # SET UP OUTPUT TEXT FILE for base and GC species
    outfile_txt = path_out + 'FINN'+finnver+'_'+simid + '_GEOSCHEM_'+sdate_emis+'_c'+todaydate+'.txt'
    print('Writing output to: ', outfile_txt)
    with open(outfile_txt, 'w') as ilun_out:
        species_list = ','.join(ef_species)
        vocs_list = ', ' + ','.join(vocnames) if nvocs_split > 0 else ''
        
        ilun_out.write('DAY,POLYID,FIREID,GENVEG,LATI,LONGI,AREA,BMASS,' + species_list + vocs_list + '\n')
        
        print('Calculating emissions... ')
        
        for ifire in range(ngood):
            emis_spec = np.zeros(nspec)
            emis_vocs = np.zeros(nvocs_split) if nvocs_split > 0 else np.array([])
            
            itype = np.where(ef_genveg == em_genveg[ifire])[0]
            if len(itype) == 0:
                ilun_log.write(f"no EF for this genveg: {em_genveg[ifire]}\n")
                continue  # skipfire4
            itype = itype[0]
            
            for ispec in range(nspec):
                if mws[ispec] != 1.:
                    emis_spec[ispec] = emisfac[itype, ispec] * 1.e-3 * em_area[ifire] * em_bmass[ifire] / (mws[ispec] * 1.e-3)
                else:
                    emis_spec[ispec] = emisfac[itype, ispec] * 1.e-3 * em_area[ifire] * em_bmass[ifire]
            
            # Calculate emissions for VOCs (moles) as fraction of NMOC (kg-species)
            if nvocs_split > 0:
                igen = np.where(genveg_voc == em_genveg[ifire])[0]
                if len(igen) > 0:
                    for ivoc in range(nvocs_split):
                        emis_vocs[ivoc] = voc_fraction[ivoc, igen[0]] * emis_spec[ind_NMOC]
                else:
                    emis_vocs[:] = 0.
            
            # Write each fire to text file
            output_line = f"{em_jday[ifire]},{em_fireid[ifire]},{em_polyid[ifire]},"
            output_line += f"{em_genveg[ifire]},{em_lat[ifire]:.3f},{em_lon[ifire]:.3f},"
            output_line += f"{em_area[ifire]:.3e},{em_bmass[ifire]:.3e},"
            output_line += ','.join([f"{x:.3e}" for x in emis_spec])
            
            if nvocs_split > 0:
                output_line += ',' + ','.join([f"{x:.3e}" for x in emis_vocs])
            
            ilun_out.write(output_line + '\n')
    
    t1 = time.time() - t0
    minutes = int(t1 // 60)
    seconds = int(t1 % 60)
    print(f'Running time: {minutes}:{seconds:02d}')
    print('Completed at: ', time.ctime())
    ilun_log.write(f'Running time: {minutes}:{seconds:02d}\n')
    ilun_log.write(f'Completed at: {time.ctime()}\n')
    
    ilun_log.close()
    ilun_gv.close()

def finn2_calc_emissions_v25():
    """
    Main program to process multiple files
    """
    today = datetime.now()
    todaystr = today.strftime('%Y%m%d')  # YYYYMMDD
    
    # for year in range(2012, 2021):
    # for year in range(2002, 2008):
    for year in range(2008, 2021):
        syr = f"{year}"
        path_in = '/data14a/FINN/processed_fires_finn2.5/'
        # file_in = path_in + 'fires_modvrs_merged_' + syr + '.csv'
        file_in = path_in + 'fires_mod_merged_' + syr + '.csv'
        
        # simid = 'modvrs_v2.5'
        simid = 'mod'
        
        path_out = '/data14a/FINN/finnv2.5/emissions/'
        
        print(f'--------- Starting processing of {year} {simid} ---------')
        
        x_finn2_calc_emissions_v2_5(file_in, simid, year, syr, todaystr, path_out)

if __name__ == "__main__":
    finn2_calc_emissions_v25()