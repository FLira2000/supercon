def prob_ratios(element):
    elementList = list(element.keys())
    prob_ratios = [ ratio/sum(list( element.values() )) for ratio in list(element.values()) ]
    chem_ratios = dict(zip(elementList, prob_ratios))
    return chem_ratios

print(prob_ratios(dict({'Na':1, 'Cl':1})))