class Small():
    def __init__(self):
        self.abbr = {'airplane':                    'AP', 
                    'bridge':                       'BR',
                    'storage-tank':                 'ST', 
                    'ship':                         'SH', 
                    'swimming-pool':                'SP', 
                    'vehicle':                      'VE', 
                    'person':                       'PE', 
                    'wind-mill':                    'WM'}

    def full2abbr(self, names):
        return [self.abbr[name] for name in names]

class DOTA():
    def __init__(self):
        self.abbr = {'harbor': 'HA', 'ship': 'SH', 'small-vehicle': 'SV', 'large-vehicle': 'LV', 'storage-tank': 'ST', 'plane': 'PL', 'soccer-ball-field': 'SBF', 'bridge': 'BR', 'baseball-diamond': 'BD', 'tennis-court': 'TC', 'helicopter': 'HC', 'roundabout': 'RA', 'swimming-pool': 'SP', 'ground-track-field': 'GTF', 'basketball-court': 'BC'}

    def full2abbr(self, names):
        return [self.abbr[name] for name in names]