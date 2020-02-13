#!/usr/bin/env python
# coding: utf-8

# ### MIE1622 - Assignment 1 - Bryan Tarras, 998853600
# 
# This notebook presents the commented and cleaned code for the submission of Assignment 1 including the modifications on part 3 for the inclusion of an alternate version to the equally weighted portfolio. Please refer to the inlcuded report (word file in submission) that details the analysis associated with the results of this notebook.

# In[23]:


# Import libraries
import pandas as pd
import numpy as np
import math

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

import cplex


# In[24]:


weights_array = [0.05]*20
print (weights_array)


# In[25]:


# Complete the following functions
def strat_buy_and_hold(x_init, cash_init, mu, Q, cur_prices):
    x_optimal = x_init
    cash_optimal = cash_init
    return (x_optimal, cash_optimal)


# In[26]:


def strat_equally_weighted_2(x_init, cash_init, mu, Q, cur_prices):
    
    first_prices = [46.759997999999996, 15.36, 33.23, 523.373108, 18.274296, 50.169998, 65.790001, 46.959999, 109.330002, 162.059998, 33.869999, 27.610001, 17.9, 36.360001, 2.67, 20.559999, 20.129999, 308.519989, 38.709999, 40.459999]
    if cur_prices[0] == first_prices[0]:
        weights_array = [0.05]*20
        x_optimal = np.dot(weights_array, np.dot(x_init,cur_prices))
        x_optimal = np.divide(x_optimal, cur_prices) 
        x_optimal = np.floor(x_optimal)
        
        #Compute the net_change in positions from the min_variance reccomended positions to the last positions 
        net_change = x_init-x_optimal
    
        #Calculate the change in price associated with this change
        net_change_price = np.dot(net_change,cur_prices)
    
        #Set cash account equal to the sum of the net of the price changes
        cash_optimal = np.sum(net_change_price)
        #print (cash_optimal)
    
        #Add previous cash_optimal value to the current cash_optimal value (linking previous periods cash amount to this period)
        cash_optimal = cash_optimal + cash_init
    
    
        ##################### CASH VALIDATION SECTION #####################
    
        #What to do if Cash_Optimal is 0?
        #Revert as many times as possible most bought current positions to the previous position until Cash Account > 0
        #Doing so would reduce the biggest subtraction from your cash_account, thus bringing it to positive the quickest
        counter = 0
        loop_break = 0
        if cash_optimal < 0:
            #Loop Condition Variable
            while loop_break == 0:
                #Sort positions based on greatest greatest change (most negative) to least (most positive)
                #since net_change is defined as intial_positions - current_suggested_positions, the most bought assets will
                #be the most negative and the one's we want to rectify first
                position_sort = np.argsort(net_change)
                #update cash_account such that reverted positions have their cash value of the transaction added back to the 
                #cash account
                cash_optimal = cash_optimal + np.absolute(net_change[position_sort[counter]]*cur_prices[position_sort[counter]])
                #Update the position array with the new positions
                x_optimal[position_sort[counter]] = x_init[position_sort[counter]]
                #Update counter so if cash_optimal is still negative the loop will continue to the next most bought asset
                #and revert its position
                counter = counter + 1
                #loop break condition is when cash_optimal is greater than 0
                if cash_optimal > 0:
                    loop_break = 1
    
        #Recalculate Net-Change as we have potentially changed the positions based on the above loop
        net_change = x_init-x_optimal
    
        #Transaction costs for the required transactions above to get to an equally weighted portfolio
        transaction_cost = np.sum(np.dot(np.absolute(net_change),cur_prices)*0.005)
        cash_optimal = cash_optimal - transaction_cost
    
        #Cash account may now be negative due to the addition of transaction costs, so this needs to be corrected
        #Similar to above revert most bought position to previous position and repeat until cash_optimal > 0
        counter_2 = 0
        loop_break_2 = 0
        if cash_optimal < 0:
            while loop_break_2 == 0:
                position_sort_2 = np.argsort(net_change)
                cash_optimal = cash_optimal + np.absolute(net_change[position_sort_2[counter_2]]*cur_prices[position_sort_2[counter_2]]*1.005)
                x_optimal[position_sort_2[counter_2]] = x_init[position_sort_2[counter_2]]
                counter_2 = counter_2 + 1
                if cash_optimal > 0:
                    loop_break_2 = 1
    
        #Next up Optimize the Cash Account as reverting positions means we have massive values in the Cash Account
        #this is obviously undesirable as it could be invested in
    
        #First Possible Scenario, Cash Account is positive and no adjustments to the positions from min_variance 
        #reccomended positions was required
        if (cash_optimal > 0 and counter == 0 and counter_2 == 0):
            extra_purchase = np.floor(np.divide(cash_optimal,(np.amin(cur_prices)*1.005)))
            #need to find the position of the min price asset and add the extra purchase to this number of shares
            position = np.argmin(cur_prices)
            x_optimal[position] = x_optimal[position] + extra_purchase
            cash_optimal = cash_optimal - np.dot(extra_purchase,np.amin(cur_prices))
            #...Need to add in transaction costs here as well...
            transaction_cost_3 = np.sum(np.dot(extra_purchase,cur_prices[position])*0.005)
            cash_optimal = cash_optimal - transaction_cost_3
    
        #Second Possible Scenario, Cash Account is positive and adjustments to the positions from min_variance
        #reccomended positions was required as the price of shares bought > price of shares sold
        elif (cash_optimal > 0 and counter > 0 and counter_2 < 1):        
            #Look to invest as many times as possible into the asset's position that was reverted back to previous position
            #in order to allow for cash positive account. Doing so preserves the reccomended positions as much as possible
            extra_purchase = np.floor(cash_optimal/((cur_prices[position_sort[counter-1]])*1.005))
            #Add these extra purchases to the array holding the positions
            x_optimal[position_sort[counter-1]] = x_optimal[position_sort[counter-1]] + extra_purchase
            #Update cash account such that the extra positions and their associated transaction costs are subtracted from
            #the cash account
            cash_optimal = cash_optimal - extra_purchase*cur_prices[position_sort[counter-1]]*1.005
    
        #Third Possible Scenario, Cash Account is positive and adjustments to the positions from min_variance
        #reccomended positions was required as the transaction fee of 0.5% for all transactions resulted in negative
        #cash account
        elif (cash_optimal > 0 and counter_2 > 0 and counter < 1):
            #Look to invest as many times as possible into the asset's position that was reverted back to previous position
            #in order to allow for cash positive account. Doing so preserves the reccomended positions as much as possible
            extra_purchase = np.floor(cash_optimal/((cur_prices[position_sort_2[counter_2-1]])*1.005))
            #Add these extra purchases to the array holding the positions
            x_optimal[position_sort_2[counter_2-1]] = x_optimal[position_sort_2[counter_2-1]] + extra_purchase
            #Update cash account such that the extra positions and their associated transaction costs are subtracted from
            #the cash account
            cash_optimal = cash_optimal - extra_purchase*cur_prices[position_sort_2[counter_2-1]]*1.005
    else:
        x_optimal = x_init
        cash_optimal = cash_init
        
    return (x_optimal, cash_optimal)


# In[27]:


def strat_equally_weighted(x_init, cash_init, mu, Q, cur_prices):
    weights_array = [0.05]*20
    x_optimal = np.dot(weights_array, np.dot(x_init,cur_prices))
    x_optimal = np.divide(x_optimal, cur_prices) 
    x_optimal = np.floor(x_optimal)
    
    #Compute the net_change in positions from the min_variance reccomended positions to the last positions 
    net_change = x_init-x_optimal
    
    #Calculate the change in price associated with this change
    net_change_price = np.dot(net_change,cur_prices)
    
    #Set cash account equal to the sum of the net of the price changes
    cash_optimal = np.sum(net_change_price)
    #print (cash_optimal)
    
    #Add previous cash_optimal value to the current cash_optimal value (linking previous periods cash amount to this period)
    cash_optimal = cash_optimal + cash_init
    
    
    ##################### CASH VALIDATION SECTION #####################
    
    #What to do if Cash_Optimal is 0?
    #Revert as many times as possible most bought current positions to the previous position until Cash Account > 0
    #Doing so would reduce the biggest subtraction from your cash_account, thus bringing it to positive the quickest
    counter = 0
    loop_break = 0
    if cash_optimal < 0:
        #Loop Condition Variable
        while loop_break == 0:
            #Sort positions based on greatest greatest change (most negative) to least (most positive)
            #since net_change is defined as intial_positions - current_suggested_positions, the most bought assets will
            #be the most negative and the one's we want to rectify first
            position_sort = np.argsort(net_change)
            #update cash_account such that reverted positions have their cash value of the transaction added back to the 
            #cash account
            cash_optimal = cash_optimal + np.absolute(net_change[position_sort[counter]]*cur_prices[position_sort[counter]])
            #Update the position array with the new positions
            x_optimal[position_sort[counter]] = x_init[position_sort[counter]]
            #Update counter so if cash_optimal is still negative the loop will continue to the next most bought asset
            #and revert its position
            counter = counter + 1
            #loop break condition is when cash_optimal is greater than 0
            if cash_optimal > 0:
                loop_break = 1
    
    #Recalculate Net-Change as we have potentially changed the positions based on the above loop
    net_change = x_init-x_optimal
    
    #Transaction costs for the required transactions above to get to an equally weighted portfolio
    transaction_cost = np.sum(np.dot(np.absolute(net_change),cur_prices)*0.005)
    cash_optimal = cash_optimal - transaction_cost
    
    #Cash account may now be negative due to the addition of transaction costs, so this needs to be corrected
    #Similar to above revert most bought position to previous position and repeat until cash_optimal > 0
    counter_2 = 0
    loop_break_2 = 0
    if cash_optimal < 0:
        while loop_break_2 == 0:
            position_sort_2 = np.argsort(net_change)
            cash_optimal = cash_optimal + np.absolute(net_change[position_sort_2[counter_2]]*cur_prices[position_sort_2[counter_2]]*1.005)
            x_optimal[position_sort_2[counter_2]] = x_init[position_sort_2[counter_2]]
            counter_2 = counter_2 + 1
            if cash_optimal > 0:
                loop_break_2 = 1
    
    #Next up Optimize the Cash Account as reverting positions means we have massive values in the Cash Account
    #this is obviously undesirable as it could be invested in
    
    #First Possible Scenario, Cash Account is positive and no adjustments to the positions from min_variance 
    #reccomended positions was required
    if (cash_optimal > 0 and counter == 0 and counter_2 == 0):
        extra_purchase = np.floor(np.divide(cash_optimal,(np.amin(cur_prices)*1.005)))
        #need to find the position of the min price asset and add the extra purchase to this number of shares
        position = np.argmin(cur_prices)
        x_optimal[position] = x_optimal[position] + extra_purchase
        cash_optimal = cash_optimal - np.dot(extra_purchase,np.amin(cur_prices))
        #...Need to add in transaction costs here as well...
        transaction_cost_3 = np.sum(np.dot(extra_purchase,cur_prices[position])*0.005)
        cash_optimal = cash_optimal - transaction_cost_3
    
    #Second Possible Scenario, Cash Account is positive and adjustments to the positions from min_variance
    #reccomended positions was required as the price of shares bought > price of shares sold
    elif (cash_optimal > 0 and counter > 0 and counter_2 < 1):        
        #Look to invest as many times as possible into the asset's position that was reverted back to previous position
        #in order to allow for cash positive account. Doing so preserves the reccomended positions as much as possible
        extra_purchase = np.floor(cash_optimal/((cur_prices[position_sort[counter-1]])*1.005))
        #Add these extra purchases to the array holding the positions
        x_optimal[position_sort[counter-1]] = x_optimal[position_sort[counter-1]] + extra_purchase
        #Update cash account such that the extra positions and their associated transaction costs are subtracted from
        #the cash account
        cash_optimal = cash_optimal - extra_purchase*cur_prices[position_sort[counter-1]]*1.005
    
    #Third Possible Scenario, Cash Account is positive and adjustments to the positions from min_variance
    #reccomended positions was required as the transaction fee of 0.5% for all transactions resulted in negative
    #cash account
    elif (cash_optimal > 0 and counter_2 > 0 and counter < 1):
        #Look to invest as many times as possible into the asset's position that was reverted back to previous position
        #in order to allow for cash positive account. Doing so preserves the reccomended positions as much as possible
        extra_purchase = np.floor(cash_optimal/((cur_prices[position_sort_2[counter_2-1]])*1.005))
        #Add these extra purchases to the array holding the positions
        x_optimal[position_sort_2[counter_2-1]] = x_optimal[position_sort_2[counter_2-1]] + extra_purchase
        #Update cash account such that the extra positions and their associated transaction costs are subtracted from
        #the cash account
        cash_optimal = cash_optimal - extra_purchase*cur_prices[position_sort_2[counter_2-1]]*1.005
    return (x_optimal, cash_optimal)


# In[28]:


def strat_min_variance(x_init, cash_init, mu, Q, cur_prices):
    
    #Define Cplex
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    
    n = len(x_init)
    c  = [0.0] * n
    lb = [0.0] * n
    ub = [1.0] * n
    
    A = []
    for k in range(n):
        A.append([[0],[1.0]])

    var_names = ["w_%s" % i for i in range(1,n+1)]
    
    #Right hand constraint is equal to 1
    #Lower bound handles the case of weights needing to be greater than 0
    cpx.linear_constraints.add(rhs=[1.0], senses="E")

    cpx.variables.add(obj=c, lb=lb, ub=ub, columns=A, names=var_names)
    
    Qmat = [[list(range(n)), list(2*Q[k,:])] for k in range(n)]
    
    cpx.objective.set_quadratic(Qmat)
    
    cpx.parameters.threads.set(4)
    
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    
    cpx.solve()
    
    #Get Solution Weights
    w_cur = cpx.solution.get_values()
    
    #Store in an weights array
    weights_array = [0.00]*20
    weights_array = w_cur
    weights_array = np.asarray(weights_array)
    
    #Compute the positions of the assets in the portfolio based on the Minimum Variance weights identified above
    x_optimal = weights_array*np.dot(x_init,cur_prices)
    x_optimal = np.divide(x_optimal, cur_prices) 
    x_optimal = np.floor(x_optimal)
    
    #Compute the net_change in positions from the min_variance reccomended positions to the last positions 
    net_change = x_init-x_optimal
    
    #Calculate the change in price associated with this change
    net_change_price = np.dot(net_change,cur_prices)
    
    #Set cash account equal to the sum of the net of the price changes
    cash_optimal = np.sum(net_change_price)
    #print (cash_optimal)
    
    #Add previous cash_optimal value to the current cash_optimal value (linking previous periods cash amount to this period)
    cash_optimal = cash_optimal + cash_init
    
    
    ##################### CASH VALIDATION SECTION #####################
    
    #What to do if Cash_Optimal is 0?
    #Revert as many times as possible most bought current positions to the previous position until Cash Account > 0
    #Doing so would reduce the biggest subtraction from your cash_account, thus bringing it to positive the quickest
    counter = 0
    loop_break = 0
    if cash_optimal < 0:
        #Loop Condition Variable
        while loop_break == 0:
            #Sort positions based on greatest greatest change (most negative) to least (most positive)
            #since net_change is defined as intial_positions - current_suggested_positions, the most bought assets will
            #be the most negative and the one's we want to rectify first
            position_sort = np.argsort(net_change)
            #update cash_account such that reverted positions have their cash value of the transaction added back to the 
            #cash account
            cash_optimal = cash_optimal + np.absolute(net_change[position_sort[counter]]*cur_prices[position_sort[counter]])
            #Update the position array with the new positions
            x_optimal[position_sort[counter]] = x_init[position_sort[counter]]
            #Update counter so if cash_optimal is still negative the loop will continue to the next most bought asset
            #and revert its position
            counter = counter + 1
            #loop break condition is when cash_optimal is greater than 0
            if cash_optimal > 0:
                loop_break = 1
    
    #Recalculate Net-Change as we have potentially changed the positions based on the above loop
    net_change = x_init-x_optimal
    
    #Transaction costs for the required transactions above to get to an equally weighted portfolio
    transaction_cost = np.sum(np.dot(np.absolute(net_change),cur_prices)*0.005)
    cash_optimal = cash_optimal - transaction_cost
    
    #Cash account may now be negative due to the addition of transaction costs, so this needs to be corrected
    #Similar to above revert most bought position to previous position and repeat until cash_optimal > 0
    counter_2 = 0
    loop_break_2 = 0
    if cash_optimal < 0:
        while loop_break_2 == 0:
            position_sort_2 = np.argsort(net_change)
            cash_optimal = cash_optimal + np.absolute(net_change[position_sort_2[counter_2]]*cur_prices[position_sort_2[counter_2]]*1.005)
            x_optimal[position_sort_2[counter_2]] = x_init[position_sort_2[counter_2]]
            counter_2 = counter_2 + 1
            if cash_optimal > 0:
                loop_break_2 = 1
    
    #Next up Optimize the Cash Account as reverting positions means we have massive values in the Cash Account
    #this is obviously undesirable as it could be invested in
    
    #First Possible Scenario, Cash Account is positive and no adjustments to the positions from min_variance 
    #reccomended positions was required
    if (cash_optimal > 0 and counter == 0 and counter_2 == 0):
        extra_purchase = np.floor(np.divide(cash_optimal,(np.amin(cur_prices)*1.005)))
        #need to find the position of the min price asset and add the extra purchase to this number of shares
        position = np.argmin(cur_prices)
        x_optimal[position] = x_optimal[position] + extra_purchase
        cash_optimal = cash_optimal - np.dot(extra_purchase,np.amin(cur_prices))
        #...Need to add in transaction costs here as well...
        transaction_cost_3 = np.sum(np.dot(extra_purchase,cur_prices[position])*0.005)
        cash_optimal = cash_optimal - transaction_cost_3
    
    #Second Possible Scenario, Cash Account is positive and adjustments to the positions from min_variance
    #reccomended positions was required as the price of shares bought > price of shares sold
    elif (cash_optimal > 0 and counter > 0 and counter_2 < 1):        
        #Look to invest as many times as possible into the asset's position that was reverted back to previous position
        #in order to allow for cash positive account. Doing so preserves the reccomended positions as much as possible
        extra_purchase = np.floor(cash_optimal/((cur_prices[position_sort[counter-1]])*1.005))
        #Add these extra purchases to the array holding the positions
        x_optimal[position_sort[counter-1]] = x_optimal[position_sort[counter-1]] + extra_purchase
        #Update cash account such that the extra positions and their associated transaction costs are subtracted from
        #the cash account
        cash_optimal = cash_optimal - extra_purchase*cur_prices[position_sort[counter-1]]*1.005
    
    #Third Possible Scenario, Cash Account is positive and adjustments to the positions from min_variance
    #reccomended positions was required as the transaction fee of 0.5% for all transactions resulted in negative
    #cash account
    elif (cash_optimal > 0 and counter_2 > 0 and counter < 1):
        #Look to invest as many times as possible into the asset's position that was reverted back to previous position
        #in order to allow for cash positive account. Doing so preserves the reccomended positions as much as possible
        extra_purchase = np.floor(cash_optimal/((cur_prices[position_sort_2[counter_2-1]])*1.005))
        #Add these extra purchases to the array holding the positions
        x_optimal[position_sort_2[counter_2-1]] = x_optimal[position_sort_2[counter_2-1]] + extra_purchase
        #Update cash account such that the extra positions and their associated transaction costs are subtracted from
        #the cash account
        cash_optimal = cash_optimal - extra_purchase*cur_prices[position_sort_2[counter_2-1]]*1.005
    
    #Return optimal positions and cash value
    return (x_optimal, cash_optimal)


# In[29]:


def strat_max_Sharpe(x_init, cash_init, mu, Q, cur_prices):
    
    #Pad the Q array to get Q~
    Q_2 = np.pad(Q, ((0,1),(0,1)), 'constant')

    #Define Cplex Model
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    
    n = len(x_init)
    
    #Set constants, lower and upper bounds
    c  = [0.0] * (n+1)
    lb = [0.0] * (n+1)
    ub = [cplex.infinity] * (n+1)
    
    #Risk Free Rate (divide by 252 as rate is shown as an annual rate)
    r_rf = (1+0.025)**(1/252) - 1
    
    #Setup A array that holds linear constraints
    A = []
    for k in range(n+1):
        if (k < 20):
            A.append([[0,1],[1,((mu[k]-r_rf))]])
        else:
            #Append kappa as the last column in the sparse matrix
            A.append([[0,1],[-1,0]])

    #Setup up variable names
    var_names = ["w_%s" % i for i in range(1,n+2)]
    
    #Set RHS constraints
    cpx.linear_constraints.add(rhs=[0,1], senses="EE")

    cpx.variables.add(obj=c, lb=lb, ub=ub, columns=A, names=var_names)
    
    Qmat = [[list(range(n+1)), list(2*Q_2[k,:])] for k in range(n+1)]
    
    cpx.objective.set_quadratic(Qmat)
        
    cpx.parameters.threads.set(6)
    
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    
    #Solve optimization probelm
    cpx.solve()
    
    #Store y values in an array
    y = cpx.solution.get_values()
    
    #Kappa is the last entry in the list
    kappa = y[-1]
    #Everything else is a y value
    y_list = y[:20]
    
    #Convert list to float array
    y_array = np.asarray(y_list)
    y_array = y_array.astype(float)
    
    #Compute the w_maxsharpe values
    w_maxsharpe = y_array/kappa
    
    #Store in an weights array as per previous formatting in Min_Variance function
    weights_array = [0.00]*20
    weights_array = w_maxsharpe
    
    #Compute the positions of the assets in the portfolio based on the Minimum Variance weights identified above
    x_optimal = weights_array*np.dot(x_init,cur_prices)
    x_optimal = np.divide(x_optimal, cur_prices) 
    x_optimal = np.floor(x_optimal)
    
    #Compute the net_change in positions from the min_variance reccomended positions to the last positions 
    net_change = x_init-x_optimal
    
    #Calculate the change in price associated with this change
    net_change_price = np.dot(net_change,cur_prices)
    
    #Set cash account equal to the sum of the net of the price changes
    cash_optimal = np.sum(net_change_price)
    #print (cash_optimal)
    
    #Add previous cash_optimal value to the current cash_optimal value (linking previous periods cash amount to this period)
    cash_optimal = cash_optimal + cash_init
    
    
    ##################### CASH VALIDATION SECTION #####################
    
    #What to do if Cash_Optimal is 0?
    #Revert as many times as possible most bought current positions to the previous position until Cash Account > 0
    #Doing so would reduce the biggest subtraction from your cash_account, thus bringing it to positive the quickest
    counter = 0
    loop_break = 0
    if cash_optimal < 0:
        #Loop Condition Variable
        while loop_break == 0:
            #Sort positions based on greatest greatest change (most negative) to least (most positive)
            #since net_change is defined as intial_positions - current_suggested_positions, the most bought assets will
            #be the most negative and the one's we want to rectify first
            position_sort = np.argsort(net_change)
            #update cash_account such that reverted positions have their cash value of the transaction added back to the 
            #cash account
            cash_optimal = cash_optimal + np.absolute(net_change[position_sort[counter]]*cur_prices[position_sort[counter]])
            #Update the position array with the new positions
            x_optimal[position_sort[counter]] = x_init[position_sort[counter]]
            #Update counter so if cash_optimal is still negative the loop will continue to the next most bought asset
            #and revert its position
            counter = counter + 1
            #loop break condition is when cash_optimal is greater than 0
            if cash_optimal > 0:
                loop_break = 1
    
    #Recalculate Net-Change as we have potentially changed the positions based on the above loop
    net_change = x_init-x_optimal
    
    #Transaction costs for the required transactions above to get to an equally weighted portfolio
    transaction_cost = np.sum(np.dot(np.absolute(net_change),cur_prices)*0.005)
    cash_optimal = cash_optimal - transaction_cost
    
    #Cash account may now be negative due to the addition of transaction costs, so this needs to be corrected
    #Similar to above revert most bought position to previous position and repeat until cash_optimal > 0
    counter_2 = 0
    loop_break_2 = 0
    if cash_optimal < 0:
        while loop_break_2 == 0:
            position_sort_2 = np.argsort(net_change)
            cash_optimal = cash_optimal + np.absolute(net_change[position_sort_2[counter_2]]*cur_prices[position_sort_2[counter_2]]*1.005)
            x_optimal[position_sort_2[counter_2]] = x_init[position_sort_2[counter_2]]
            counter_2 = counter_2 + 1
            if cash_optimal > 0:
                loop_break_2 = 1
    
    #Next up Optimize the Cash Account as reverting positions means we have massive values in the Cash Account
    #this is obviously undesirable as it could be invested in
    
    #First Possible Scenario, Cash Account is positive and no adjustments to the positions from min_variance 
    #reccomended positions was required
    if (cash_optimal > 0 and counter == 0 and counter_2 == 0):
        extra_purchase = np.floor(np.divide(cash_optimal,(np.amin(cur_prices)*1.005)))
        #need to find the position of the min price asset and add the extra purchase to this number of shares
        position = np.argmin(cur_prices)
        x_optimal[position] = x_optimal[position] + extra_purchase
        cash_optimal = cash_optimal - np.dot(extra_purchase,np.amin(cur_prices))
        #...Need to add in transaction costs here as well...
        transaction_cost_3 = np.sum(np.dot(extra_purchase,cur_prices[position])*0.005)
        cash_optimal = cash_optimal - transaction_cost_3
    
    #Second Possible Scenario, Cash Account is positive and adjustments to the positions from min_variance
    #reccomended positions was required as the price of shares bought > price of shares sold
    elif (cash_optimal > 0 and counter > 0 and counter_2 < 1):        
        #Look to invest as many times as possible into the asset's position that was reverted back to previous position
        #in order to allow for cash positive account. Doing so preserves the reccomended positions as much as possible
        extra_purchase = np.floor(cash_optimal/((cur_prices[position_sort[counter-1]])*1.005))
        #Add these extra purchases to the array holding the positions
        x_optimal[position_sort[counter-1]] = x_optimal[position_sort[counter-1]] + extra_purchase
        #Update cash account such that the extra positions and their associated transaction costs are subtracted from
        #the cash account
        cash_optimal = cash_optimal - extra_purchase*cur_prices[position_sort[counter-1]]*1.005
    
    #Third Possible Scenario, Cash Account is positive and adjustments to the positions from min_variance
    #reccomended positions was required as the transaction fee of 0.5% for all transactions resulted in negative
    #cash account
    elif (cash_optimal > 0 and counter_2 > 0 and counter < 1):
        #Look to invest as many times as possible into the asset's position that was reverted back to previous position
        #in order to allow for cash positive account. Doing so preserves the reccomended positions as much as possible
        extra_purchase = np.floor(cash_optimal/((cur_prices[position_sort_2[counter_2-1]])*1.005))
        #Add these extra purchases to the array holding the positions
        x_optimal[position_sort_2[counter_2-1]] = x_optimal[position_sort_2[counter_2-1]] + extra_purchase
        #Update cash account such that the extra positions and their associated transaction costs are subtracted from
        #the cash account
        cash_optimal = cash_optimal - extra_purchase*cur_prices[position_sort_2[counter_2-1]]*1.005
        
    return (x_optimal, cash_optimal)


# In[30]:


# Input file
input_file_prices = 'Daily_closing_prices.csv'

# Read data into a dataframe
df = pd.read_csv(input_file_prices)


# In[31]:


# Convert dates into array [year month day]
def convert_date_to_array(datestr):
    temp = [int(x) for x in datestr.split('/')]
    return [temp[-1], temp[0], temp[1]]


# In[32]:


dates_array = np.array(list(df['Date'].apply(convert_date_to_array)))
data_prices = df.iloc[:, 1:].to_numpy()
dates = np.array(df['Date'])


# In[33]:


# Find the number of trading days in Nov-Dec 2014 and
# compute expected return and covariance matrix for period 1
day_ind_start0 = 0
day_ind_end0 = len(np.where(dates_array[:,0]==2014)[0])
cur_returns0 = data_prices[day_ind_start0+1:day_ind_end0,:] / data_prices[day_ind_start0:day_ind_end0-1,:] - 1
mu = np.mean(cur_returns0, axis = 0)
Q = np.cov(cur_returns0.T)


# In[34]:


# Remove datapoints for year 2014
data_prices = data_prices[day_ind_end0:,:]
dates_array = dates_array[day_ind_end0:,:]
dates = dates[day_ind_end0:]


# In[35]:


# Initial positions in the portfolio
init_positions = np.array([5000, 950, 2000, 0, 0, 0, 0, 2000, 3000, 1500, 0, 0, 0, 0, 0, 0, 1001, 0, 0, 0])


# In[36]:


# Initial value of the portfolio
init_value = np.dot(data_prices[0,:], init_positions)
print('\nInitial portfolio value = $ {}\n'.format(round(init_value, 2)))


# In[37]:


# Initial portfolio weights
w_init = (data_prices[0,:] * init_positions) / init_value


# In[38]:


print (w_init)


# In[39]:


# Number of periods, assets, trading days
N_periods = 6*len(np.unique(dates_array[:,0])) # 6 periods per year
N = len(df.columns)-1
N_days = len(dates)


# In[40]:


# Annual risk-free rate for years 2015-2016 is 2.5%
r_rf = 0.025


# In[41]:


# Number of strategies
strategy_functions = ['strat_buy_and_hold', 'strat_equally_weighted', 'strat_min_variance', 'strat_max_Sharpe', 'strat_equally_weighted_2']
strategy_names     = ['Buy and Hold', 'Equally Weighted Portfolio', 'Mininum Variance Portfolio', 'Maximum Sharpe Ratio Portfolio', 'Equally Weighted Portfolio Version 2']
#N_strat = 4  # comment this in your code
N_strat = len(strategy_functions)  # uncomment this in your code
fh_array = [strat_buy_and_hold, strat_equally_weighted, strat_min_variance, strat_max_Sharpe, strat_equally_weighted_2]


# In[42]:


portf_value = [0] * N_strat
x = np.zeros((N_strat, N_periods),  dtype=np.ndarray)
cash = np.zeros((N_strat, N_periods),  dtype=np.ndarray)


# In[43]:


all_prices = np.zeros((N_strat, N_periods),  dtype=np.ndarray)


# In[44]:


for period in range(1, N_periods+1):
    # Compute current year and month, first and last day of the period

    if dates_array[0, 0] == 15:
        cur_year  = 15 + math.floor(period/7)
    else:
        cur_year  = 2015 + math.floor(period/7)

    cur_month = 2*((period-1)%6) + 1
    day_ind_start = min([i for i, val in enumerate((dates_array[:,0] == cur_year) & (dates_array[:,1] == cur_month)) if val])
    day_ind_end = max([i for i, val in enumerate((dates_array[:,0] == cur_year) & (dates_array[:,1] == cur_month+1)) if val])
    print('\nPeriod {0}: start date {1}, end date {2}'.format(period, dates[day_ind_start], dates[day_ind_end]))
    
    # Prices for the current day
    cur_prices = data_prices[day_ind_start,:]

    # Execute portfolio selection strategies
    for strategy in range(N_strat):

        # Get current portfolio positions
        if period == 1:
            curr_positions = init_positions
            curr_cash = 0
            portf_value[strategy] = np.zeros((N_days, 1))
        else:
            curr_positions = x[strategy, period-2]
            curr_cash = cash[strategy, period-2]

        # Compute strategy
        
        x[strategy, period-1], cash[strategy, period-1] = fh_array[strategy](curr_positions, curr_cash, mu, Q, cur_prices)
        
        # Verify that strategy is feasible (you have enough budget to re-balance portfolio)
        # Check that cash account is >= 0
        # Check that we can buy new portfolio subject to transaction costs

        ###################### Insert your code here ############################
        #FOR THE TA AND PROF
            #I have put the cash validation code in each function for the sake of debugging and keeping positional changes
            #within the functions themselves. Please refer to the portion of the functions which state:
            ##################### CASH VALIDATION SECTION #####################
                    
        # Compute portfolio value
        p_values = np.dot(data_prices[day_ind_start:day_ind_end+1,:], x[strategy, period-1]) + cash[strategy, period-1]
        portf_value[strategy][day_ind_start:day_ind_end+1] = np.reshape(p_values, (p_values.size,1))
        print('  Strategy "{0}", value begin = $ {1:.2f}, value end = $ {2:.2f}'.format(strategy_names[strategy], 
                                                                                        portf_value[strategy][day_ind_start][0], portf_value[strategy][day_ind_end][0]))
      
    # Compute expected returns and covariances for the next period
    cur_returns = data_prices[day_ind_start+1:day_ind_end+1,:] / data_prices[day_ind_start:day_ind_end,:] - 1
    mu = np.mean(cur_returns, axis = 0)
    Q = np.cov(cur_returns.T)

    # Plot results
    ###################### Insert your code here ############################
    #Store all prices across the strategies analyzed
    all_prices[strategy, period-1] = cur_prices
    #Refer to cells below for the plotted graphs


# ### Plotting the Weight Allocations for Minimum Variance and Max Sharpe Ratio Portfolios

# In[45]:


#Setup a prices dataframe that stores the prices of each asset across the periods
df_prices = pd.DataFrame(all_prices[4][:], columns = ['Prices'])
df_prices.tail()


# In[46]:


#Setup a positions dataframe that holds the positions for each portfolio across each period
df_positions = pd.DataFrame(x[0][:], columns = ['Buy and Hold']) 
df_positions['Equally Weighted Portfolio'] = x[1][:]
df_positions['Mininum Variance Portfolio'] = x[2][:]
df_positions['Maximum Sharpe Ratio Portfolio'] = x[3][:]
df_positions['Equally Weighted Portfolio Version 2'] = x[4][:]
df_positions.tail()


# In[47]:


print (portf_value[4][0])


# In[48]:


#Setup a dataframe that holds Portfolio Value for each of the stratgies across the 12 periods
df_strategies = pd.DataFrame(portf_value[0][:], columns = ['Buy and Hold']) 
df_strategies['Equally Weighted Portfolio'] = portf_value[1][:]
df_strategies['Mininum Variance Portfolio'] = portf_value[2][:]
df_strategies['Maximum Sharpe Ratio Portfolio'] = portf_value[3][:]
df_strategies['Equally Weighted Portfolio Version 2'] = portf_value[4][:]
df_strategies.head()


# weights = (prices * current_positions)/Portfolio_Value

# Make a dataframe that holds the prices as period per column

# In[49]:


#Make new dataframe that holds the prices with columns representing periods and rows the assets
df_price_period = pd.DataFrame(df_prices.iloc[0,0], columns = ['Period 0'])
x = 1
for x in range (12):
    df_price_period["{} {}".format("Period", x+1)] = df_prices.iloc[x,0]

df_price_period = df_price_period.drop(['Period 0'], axis=1)

df_price_period.tail()


# ### Weight Graph for Minimum Variance

# In[50]:


#Make new dataframe that holds the Minimum Variance Portfolio weights with columns representing periods and rows the assets
df_positions_min_variance = pd.DataFrame(df_positions.iloc[0,2], columns = ['Period 0'])
x = 1
for x in range (12):
    df_positions_min_variance["{} {}".format("Period", x+1)] = df_positions.iloc[x,2]

df_positions_min_variance = df_positions_min_variance.drop(['Period 0'], axis=1)

df_positions_min_variance.tail()


# In[51]:


#Setup a weights dataframe
df_weights = pd.DataFrame(columns=['Period 1', 'Period 2', 'Period 3', 'Period 4', 'Period 5', 'Period 6', 'Period 7', 'Period 8', 'Period 9', 'Period 10', 'Period 11', 'Period 12'])
df_weights.head()


# In[52]:


#Populate the weights dataframe with values for minimum variance portfolio
y = 0
for x in range (12):
    if x == 0:
        df_weights["{} {}".format("Period", x+1)] = df_price_period["{} {}".format("Period", x+1)]*df_positions_min_variance["{} {}".format("Period", x+1)]/df_strategies['Mininum Variance Portfolio'][y]
    else:
        y = x * 42
        df_weights["{} {}".format("Period", x+1)] = df_price_period["{} {}".format("Period", x+1)]*df_positions_min_variance["{} {}".format("Period", x+1)]/df_strategies['Mininum Variance Portfolio'][y]

df_weights.head(20)


# In[53]:


#Transpose the matrix for ease of processing
df_weights = df_weights.T


# In[54]:


#I will be using Seaborn for plotting, so the use of pd.melt will be required
data_2 = pd.melt(df_weights)
data_2 = data_2.reset_index()
data_2.head()
data_2 = data_2.rename(columns={"index": "Period"})
data_2['counter'] = data_2.Period


# In[55]:


#Resetting the Period counter as the portfolios have been melted together into one coloumn
counter = 0
for x in range (len(data_2)):
    if (data_2.iloc[x,3]%12) == 0:
        data_2.iloc[x,0] = 0
        counter = 0
    else:
        counter = counter + 1
        data_2.iloc[x,0] = counter


# In[56]:


#Resize plot to make it easier to read
from matplotlib.pyplot import figure
fig_size = plt.rcParams["figure.figsize"]
print ("Current size:"), fig_size

fig_size[0] = 12
fig_size[1] = 9
print ("Current size:"), fig_size


# In[57]:


#sns.set_palette(sns.color_palette("hls"))
#Line Plot will be used for plotting the portfolios against one another
ax = sns.lineplot(x='Period', y='value', hue='variable', 
             data=data_2, legend='full')
plt.ticklabel_format(style='plain', axis='y')
plt.ticklabel_format(style='plain', axis='y')
ax.set_xlabel('Period')
ax.set_ylabel('Weight Allocation')
plt.title('Minimum Variance Portfolio Weight Allocation')
plt.show()


# ### Weight Graph for Maximum Sharpe Ratio

# In[58]:


#Make new dataframe that holds the Minimum Variance Portfolio weights with columns representing periods and rows the assets
df_positions_max_sharpe = pd.DataFrame(df_positions.iloc[0,3], columns = ['Period 0'])
x = 1
for x in range (12):
    df_positions_max_sharpe["{} {}".format("Period", x+1)] = df_positions.iloc[x,3]

df_positions_max_sharpe = df_positions_max_sharpe.drop(['Period 0'], axis=1)

df_positions_max_sharpe.tail()


# In[59]:


#Setup a weights dataframe
df_weights = pd.DataFrame(columns=['Period 1', 'Period 2', 'Period 3', 'Period 4', 'Period 5', 'Period 6', 'Period 7', 'Period 8', 'Period 9', 'Period 10', 'Period 11', 'Period 12'])
df_weights.head()


# In[60]:


#Populate the weights dataframe with values for minimum variance portfolio
y = 0
for x in range (12):
    if x == 0:
        df_weights["{} {}".format("Period", x+1)] = df_price_period["{} {}".format("Period", x+1)]*df_positions_max_sharpe["{} {}".format("Period", x+1)]/df_strategies['Maximum Sharpe Ratio Portfolio'][y]
    else:
        y = x * 42
        df_weights["{} {}".format("Period", x+1)] = df_price_period["{} {}".format("Period", x+1)]*df_positions_max_sharpe["{} {}".format("Period", x+1)]/df_strategies['Maximum Sharpe Ratio Portfolio'][y]

df_weights.head(20)


# In[61]:


#Transpose the matrix for ease of processing
df_weights = df_weights.T


# In[62]:


#I will be using Seaborn for plotting, so the use of pd.melt will be required
data_2 = pd.melt(df_weights)
data_2 = data_2.reset_index()
data_2.head()
data_2 = data_2.rename(columns={"index": "Period"})
data_2['counter'] = data_2.Period


# In[63]:


#Resetting the Period counter as the portfolios have been melted together into one coloumn
counter = 0
for x in range (len(data_2)):
    if (data_2.iloc[x,3]%12) == 0:
        data_2.iloc[x,0] = 0
        counter = 0
    else:
        counter = counter + 1
        data_2.iloc[x,0] = counter


# In[64]:


#sns.set_palette(sns.color_palette("hls"))
#Line Plot will be used for plotting the portfolios against one another
ax = sns.lineplot(x='Period', y='value', hue='variable', 
             data=data_2, legend='full')
plt.ticklabel_format(style='plain', axis='y')
plt.ticklabel_format(style='plain', axis='y')
ax.set_xlabel('Period')
ax.set_ylabel('Weight Allocation')
plt.title('Maximum Sharpe Ratio Portfolio Weight Allocation')
plt.show()


# ### Plotting The 4 Strategy Returns on One Graph

# In[65]:


#Setup dataframe that holds the returns each day per strategy
df_strategies = pd.DataFrame(portf_value[0][:], columns = ['Buy and Hold']) 
df_strategies['Equally Weighted Portfolio'] = portf_value[1][:]
df_strategies['Mininum Variance Portfolio'] = portf_value[2][:]
df_strategies['Maximum Sharpe Ratio'] = portf_value[3][:]
df_strategies['Equally Weighted Portfolio Version 2'] = portf_value[4][:]
df_strategies.tail()


# In[66]:


#I will be using Seaborn for plotting, so the use of pd.melt will be required
data=pd.melt(df_strategies)
data = data.reset_index()
data = data.rename(columns={"index": "Day"})
data['counter'] = data.Day


# In[67]:


#Resetting the Day counter as the portfolios have been melted together into one coloumn
counter = 0
for x in range (5*len(portf_value[1][:])):
    if data.iloc[x,3] == 504:
        data.iloc[x,0] = 0
        counter = 0
    elif data.iloc[x,3] == 1008:
        data.iloc[x,0] = 0
        counter = 0
    elif data.iloc[x,3] == 1512:
        data.iloc[x,0] = 0
        counter = 0
    elif data.iloc[x,3] == 2016:
        data.iloc[x,0] = 0
        counter = 0
    else:
        counter = counter + 1
        data.iloc[x,0] = counter


# In[68]:


#Line Plot will be used for plotting the portfolios against one another
ax = sns.lineplot(x='Day', y='value', hue='variable', 
             data=data)
plt.ticklabel_format(style='plain', axis='y')
plt.ticklabel_format(style='plain', axis='y')
ax.set_xlabel('Days')
ax.set_ylabel('Portfolio Value ($)')
plt.title('Portfolio Strategies Returns')
plt.show()


# In[ ]:




