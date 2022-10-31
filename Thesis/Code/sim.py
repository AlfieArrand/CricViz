# just gonna whack the new sim in here and work on it iteratively. A lot of shit nicked from the Basic one.
# Still on first innings.


import pickle 
import pandas as pd
import numpy as np
import random as rd
import math


class T20SimX:
    def __init__(self, dir, mechanism = 'Arrand'):
        """
        __init__

        Parameters
        ---

        mechanism: str in ['Basic', 'Davis', 'Arrand']
            Distribution model for batting outcomes. 

        Returns
        ---

        None
        """
        self.mech = mechanism

        self.X_iowj = None
        self.Y_iowj = None
        
        #extract these from input directories
        f = open(dir, 'rb')
        (self.batter_idx, self.bowler_idx, self.outcomes, self.deliveries,
        self.p_fwicket, self.p_fextras, self.d_fextras, self.d_wides, self.d_nbs,
        self.m_freehit_j, self.m_nb_j, self.Tau_owj, self.Xi_od, self.p_i70j, 
        self.q_i70j, self.rho_i1d, self.avg70j) = pickle.load(f)
        f.close()

        self.bowlers = list(self.q_i70j.keys())
        self.bats = list(self.p_i70j.keys())

        self.p_fwicket = [0.001961639058413252,
 0.0028353326063249727,
 0.0024250440917107582,
 0.0035343494588027393,
 0.0022104332449160036,
 0.0033127208480565372,
 0.003324468085106383,
 0.0033429908624916425,
 0.004008016032064128,
 0.0022326412145568207,
 0.0024741340530814214,
 0.0024780355936021626,
 0.0029220049449314453,
 0.004287971112615662,
 0.003819366434509099,
 0.006310570205093531,
 0.0049493813273340835,
 0.007898894154818325,
 0.009768287142208088,
 0.037572928821470244]
        
        self.Tau_owj = self.Tau_owj ** 1
        self.Tau_owj_bat = self.Tau_owj ** 1

    def get_distn(self, batsman, bowler, wickets = 0, over = 7, innings = 1):

        m_distn = (self.avg70j * self.Tau_owj[over-1,wickets,:])/np.sum(self.avg70j * self.Tau_owj[over-1,wickets,:])

        if batsman in self.bats:
            bat_distn = (self.p_i70j[batsman] * self.Tau_owj_bat[over-1,wickets,:])/np.sum(self.p_i70j[batsman] * self.Tau_owj_bat[over-1,wickets,:])
        else:
            bat_distn = m_distn

        if bowler in self.bowlers:
            bowl_distn = (self.q_i70j[bowler] * self.Tau_owj[over-1,wickets,:])/np.sum(self.q_i70j[bowler] * self.Tau_owj[over-1,wickets,:])
        else:
            bowl_distn = m_distn

        distn = (bat_distn + bowl_distn - m_distn)

        return distn

    def sim_fairball(self, batsman, bowler, wickets = 0, over = 7, innings = 1):

        distn = (self.get_distn(batsman, bowler, wickets, over, innings)) #* np.array([1.12342569, 0.9513847 , 0.88677233, 0.80167598, 0.93503737,0.73529412, 0.89628009, 1.14851485]))
        
        if over < 7:
            distn *= np.array([1.24684967, 0.77482206, 0.72245287, 1.03889716, 1.00004945, 2.2038279 , 0.71157346, 1.39323055])

        distn = (distn/sum(distn)).cumsum()

        p = rd.random()
            
        i = 0
        while distn[i] < p:
            i += 1

        return self.outcomes[i]

    def sim_freehit(self, batsman, bowler):
        f = self.m_freehit_j
        distn = f * self.get_distn(batsman, bowler)
        distn = (distn/np.sum(distn)).cumsum()

        p = rd.random()
            
        i = 0
        while distn[i] < p:
            i += 1

        return self.outcomes[i]

    def sim_noball(self, batsman, bowler, wickets = 0, over = 7, innings = 1):

        p_ = rd.random()
                
        l = 0
        while self.d_nbs.cumsum()[l] < p_:
            l += 1

        nbs = int(self.outcomes[l])

        if nbs == 1:
            n = self.m_nb_j
            
            distn = n * self.get_distn(batsman, bowler, wickets, over, innings)
            distn = (distn/np.sum(distn)).cumsum()

            p = rd.random()

            i = 0
            while distn[i] < p:
                i += 1

            runs = self.outcomes[i]
        
        else:
            runs = '0'

        return nbs, runs

    def delivery(self, bowler, over):
        if bowler in self.bowlers:
            t = (self.rho_i1d[bowler]*self.Xi_od[over-1,:])
        else:
            t = np.mean(list(self.rho_i1d.values()),axis=0)*self.Xi_od[over-1,:]
        distn = (t/np.sum(t)).cumsum()
        p = rd.random()
                
        i = 0
        while distn[i] < p:
            i += 1

        return self.deliveries[i]

    def sim_over(self, striker, nonstriker, batsmen_rem, bowler, wickets = 0, over = 7, innings = 1):

        fair_balls = 0
        over_ = []
        freehit = False
        while fair_balls < 6 and wickets < 10:

            facing = striker
            delivery = self.delivery(bowler, over)

            if delivery == 'fair':
                nbs = 0
                wides = 0
                if freehit:
                    runs = self.sim_freehit(striker, bowler)
                    freehit = False
                else:
                    runs = self.sim_fairball(striker, bowler, wickets, over, innings)
                if runs == 'D':
                    wickets += 1
                    try: 
                        striker = batsmen_rem[0]
                        batsmen_rem.pop(0)
                    except IndexError:
                        over_.append([innings,over,wickets,facing,bowler,delivery,wides,nbs,runs,0,False])
                        break
                elif runs in ['1','3','5']:
                    striker, nonstriker = nonstriker, striker

                fair_balls += 1

            elif delivery == 'wide':
                
                p_ = rd.random()
                
                i = 0
                while self.d_wides.cumsum()[i] < p_:
                    i += 1

                wides = int(self.outcomes[i])
                nbs = 0
                runs = '0'

                if wides%2 == 0:
                    striker, nonstriker = nonstriker, striker

            else:
                wides = 0
                nbs, runs = self.sim_noball(striker, bowler, wickets , over, innings)
                freehit = True
            
            if runs == '0':
                if rd.random() < self.p_fextras: 
                    fextras = 1
                    striker, nonstriker = nonstriker, striker
                else:
                    fextras = 0
            else:
                fextras = 0

            if runs not in ['4', '6', 'D']:
                if rd.random() < self.p_fwicket[over-1]:
                    fwicket = True
                    wickets += 1
                    try: 
                        striker = batsmen_rem[0]
                        batsmen_rem.pop(0)
                    except IndexError:
                        over_.append([innings,over,wickets,facing,bowler,delivery,wides,nbs,runs,fextras,fwicket])
                        break
                else:
                    fwicket = False
            else:
                fwicket = False

            over_.append([innings,over,wickets,facing,bowler,delivery,wides,nbs,runs,fextras,fwicket])
            
        return over_, batsmen_rem, striker, nonstriker

    def howout(self, inns_df, bat):
        last_ball = inns_df[inns_df.striker == bat].iloc[-1,:]

        if last_ball.fwicket:
            return 'run out'
        elif last_ball.runs == 'D':
            return 'b ' + last_ball.bowler
        else: 
            return 'not out'

    def runs_conc(self, inns_df, bowler):
        """
        Computes runs conceded by bowler.
        """

        return np.array(list(map(int, inns_df[inns_df.bowler == bowler][inns_df.runs != 'D'].runs))).sum() + inns_df[inns_df.bowler == bowler].wides.sum() + inns_df[inns_df.bowler == bowler].nbs.sum()

    def sim_innings1(self, team_name, batting_order, bowling_order, res = 'ball-by-ball', score = False):
        """
        Simulates a 20 over first innings, given the order of batsman and bowlers. 

        Parameters
        ---

        batting_order: list
            A list of names of the order of batsmen in the batting side. 

        bowling_order: list 
            A list of length 20 identifying the bowler for each over in the innings. 

        scorecard: bool
            If True, the method returns the scorecard, for both batsmem and bowlers, instead of the ball-by-ball report.

        Returns
        ---

        inns_df: DataFrame 
            A pandas DataFrame giving the ball-by-ball outcome of the innings.

        """

        innings, batrem, striker, nonstriker = self.sim_over(batting_order[0], batting_order[1], batting_order[2:], bowling_order[0], 0,1,1)

        while innings[-1][2] < 10:
            for i in range(2,21):
                over, batrem, striker, nonstriker = self.sim_over(nonstriker, striker, batrem, bowling_order[i-1], innings[-1][2],i,1)
                innings += over
            break

        inns_df = pd.DataFrame(innings, columns = ['innings','over','wickets','striker','bowler','delivery','wides','nbs','runs','fextras','fwicket'])

        if score: 
            runs = np.sum(list(map(int, inns_df[inns_df.runs != 'D'].runs))) + inns_df.wides.sum() + inns_df.nbs.sum() + inns_df.fextras.sum()
            
            if list(inns_df.wickets)[-1] == 10:
                overs = list(inns_df.over)[-1] - 1
                balls = len(inns_df[inns_df.over == overs][inns_df.delivery == 'fair'])
            else:
                overs, balls = 20, 0

            print(team_name + ': ' + str(runs) + ' - ' + str(list(inns_df.wickets)[-1]) + ' from ' + str(overs) + '.' + str(balls))

        if res == 'scorecard': # batter, howout, runs, balls, 4s, 6s
            bat_res = [[bat, self.howout(inns_df, bat), np.array(list(map(int, inns_df[inns_df.striker == bat][inns_df.runs != 'D'].runs))).sum(), inns_df[inns_df.striker == bat][inns_df.delivery != 'wide'].shape[0], list(inns_df[inns_df.striker == bat].runs).count('4'), list(inns_df[inns_df.striker == bat].runs).count('6')] for bat in batting_order[:len(set(inns_df.striker))]]
            # bowler, overs, runs, wickets, economy, 
            bowl_res = [[bowler, bowling_order.count(bowler),self.runs_conc(inns_df, bowler), list(inns_df[inns_df.bowler == bowler].runs).count('D'), self.runs_conc(inns_df, bowler)/bowling_order.count(bowler)] for bowler in set(bowling_order)]
            return pd.DataFrame(bat_res, columns = ['batter', 'howout', 'runs', 'balls', '4s', '6s']), pd.DataFrame(bowl_res, columns = ['bowler', 'overs', 'runs', 'wickets', 'economy'])
        elif res == 'runs': #just returns the runs
            return np.sum(list(map(int, inns_df[inns_df.runs != 'D'].runs))) + inns_df.wides.sum() + inns_df.nbs.sum() + inns_df.fextras.sum()
        elif res == 'wick':
            wicks =  [0] + [list(inns_df[inns_df.over == o].wickets)[-1] for o in sorted(list(set(inns_df.over)))]
            # if (20 in set(inns_df.over)):
            #     wicks += [(list(inns_df[inns_df.over == 20].wickets)[-1])]
            return wicks
        else:
            return inns_df

    def sim_innings1_fromstate(self, striker, nonstriker, batsmen_rem, over, wickets, bowling_rem):
        
        innings = [['Resuming', 'with', wickets, 'wickets', 'down', 'and', over - 1, 'overs', 'gone.', ' ', 'Play!']]
        
        striker, nonstriker = nonstriker, striker
        while innings[-1][2] < 10:
            for i in range(over,21):
                over_, batsmen_rem, striker, nonstriker = self.sim_over(nonstriker, striker, batsmen_rem, bowling_rem[i-over-1], innings[-1][2],i,1)
                innings += over_
            break
        
        inns_df =  pd.DataFrame(innings, columns = ['innings','over','wickets','striker','bowler','delivery','wides','nbs','runs','fextras','fwicket'])

        return inns_df


