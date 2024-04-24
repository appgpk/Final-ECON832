import delimited "/Users/pengdewendecarmelmarief.zagre/Downloads/Final/DataToExploit.csv", clear

gen ratio = .

levelsof gameid, local(gameIDs)
levelsof subjid, local(participants)
foreach gameID of local gameIDs {
    foreach participant of local participants {
        forvalues i = 1/5 {
            count if gameid == `gameID' & subjid == `participant' & b == 1 & block == `i'
            local count_b_equal_1 = r(N)
            replace ratio = `count_b_equal_1'/5 if gameid == `gameID' & subjid == `participant' & block == `i'
        }
    }
}

encode gender, generate (Gender)
drop gender

label list Gender
* 1 F
* 2 M
label values Gender


encode location, generate (Location)
drop location
label list Location
* 1 Rehovot
* 2 Technion
label values Location

encode condition, generate (Condition)
drop condition
label list Condition
* 1 ByProb
label values Condition


encode lotshapea, generate (ShapeA)
drop lotshapea
label list ShapeA
* 1 -
* 2 L-skew
* 3 R-skew
* 4 Symm
label values ShapeA

encode lotshapeb, generate (ShapeB)
drop lotshapeb
label list ShapeB
* 1 -
* 2 L-skew
* 3 R-skew
* 4 Symm
label values ShapeB


encode button, generate (Button)
drop button
label list Button
* 1 L
* 2 R
label values Button
gen diffev = eva-evb


sort subjid gameid block trial
by subjid gameid block: gen change_reactiontime = rt[_n]-rt[_n-1]
replace change_reactiontime = (change_reactiontime[_n+1] + change_reactiontime[_n+2]+change_reactiontime[_n+3]+change_reactiontime[_n+4])/4 if change_reactiontime ==.


sort subjid gameid block trial
by subjid gameid block: gen prev_B = b[_n-1] if _n > 1
by subjid gameid block: gen switch_count = (b != prev_B) & !missing(prev_B)
by subjid gameid block: gen behavior_change = sum(switch_count)

gen behavior_change_af = 1 if feedback == 1 & behavior_change[_n]!= behavior_change[_n-1]
replace behavior_change_af =0 if behavior_change_af==.


keep gameid subjid b age set ha pha la lotnuma hb phb lb lotnumb amb corr block  Gender Location Condition ShapeA ShapeB order trial payoff forgone hapwa lapwa hapwb lapwb eva evb diffev skewa skewb dominance 
export delimited using "/Users/pengdewendecarmelmarief.zagre/Downloads/Final/finaldata1.csv", replace


keep gameid subjid b age set ha pha la lotnuma hb phb lb lotnumb amb corr block  Gender Location Condition ShapeA ShapeB order trial payoff forgone hapwa lapwa hapwb lapwb eva evb diffev skewa skewb dominance relativeorder change_reactiontime behavior_change_af feedback rt
export delimited using "/Users/pengdewendecarmelmarief.zagre/Downloads/Final/finaldata2.csv", replace


////////////////////////////////////////////////////////////////////////////////
///////This part of the code is to be run after running the julia fille/////////
////////////////////////////////////////////////////////////////////////////////
import delimited "/Users/pengdewendecarmelmarief.zagre/Downloads/Final/datafinal.csv", clear
gen predictedratio1 = .
gen predictedratio2 = .

levelsof gameid, local(gameIDs)
levelsof subjid, local(participants)
foreach gameID of local gameIDs {
    foreach participant of local participants {
        forvalues i = 1/5 {
            count if gameid == `gameID' & subjid == `participant' & predicted_class1 == 1 & block == `i'
            local count_equal_1 = r(N)
            replace predictedratio1 = `count_equal_1'/5 if gameid == `gameID' & subjid == `participant' & block == `i'
        }
    }
}


foreach gameID of local gameIDs {
    foreach participant of local participants {
        forvalues i = 1/5 {
            count if gameid == `gameID' & subjid == `participant' & predicted_class2 == 1 & block == `i'
            local count_equal_1 = r(N)
            replace predictedratio2 = `count_equal_1'/5 if gameid == `gameID' & subjid == `participant' & block == `i'
        }
    }
}

keep subjid gameid predictedratio* block
collapse (mean) predictedratio1 predictedratio2, by(subjid gameid block)
reshape wide predictedratio*, i(subjid gameid) j(block) 

rename predictedratio11 B11
rename predictedratio12 B12
rename predictedratio13 B13
rename predictedratio14 B14
rename predictedratio15 B15

rename predictedratio21 B21
rename predictedratio22 B22
rename predictedratio23 B23
rename predictedratio24 B24
rename predictedratio25 B25
save "/Users/pengdewendecarmelmarief.zagre/Downloads/Final/datafinal.dta", replace

import delimited "/Users/pengdewendecarmelmarief.zagre/Downloads/Final/Data-to-predict-Track-2.csv", clear

merge m:m subjid gameid using "/Users/pengdewendecarmelmarief.zagre/Downloads/Final/datafinal.dta"
drop _merge

export delimited using "/Users/pengdewendecarmelmarief.zagre/Downloads/Final/finaldatapredict.csv", replace

