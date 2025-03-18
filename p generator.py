import itertools
def getAllowedMoves(df, utils):
    B = df['Block'].unique()
    P_set = set()
    
    weekend_days = {5, 6}      
    allowed_weekend_moves = {4, 5, 6}  

    for b1, b2 in itertools.product(B, repeat=2):
        o1 = utils.getRoom(b1)
        o2 = utils.getRoom(b2)
        
        if abs(o1 - o2) > 10:
            continue

        d1 = b1[1]
        d2 = b2[1]
        
        if (d1 in weekend_days or d2 in weekend_days) and not (d1 in allowed_weekend_moves and d2 in allowed_weekend_moves):
            continue
        
        P_set.add((b1, b2))
    
    return list(P_set)