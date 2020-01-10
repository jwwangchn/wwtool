class Small():
    def __init__(self):
        self.abbr = {'airplane':                    'AP', 
                    'bridge':                       'BR',
                    'storage-tank':                 'ST', 
                    'ship':                         'SH', 
                    'swimming-pool':                'SP', 
                    'tennis-court':                 'TC', 
                    'vehicle':                      'VE', 
                    'person':                       'PE', 
                    'harbor':                       'HA', 
                    'wind-mill':                    'WM'}

    def full2abbr(self, names):
        return [self.abbr[name] for name in names]