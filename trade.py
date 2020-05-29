class Trade:
	def __init__(self, instrument, type, units, row, stop_loss, take_profit):
		self.Type = type								# BUY or SELL
		self.Units = units								# trade ammount
		self.SL = stop_loss								# stop loss
		self.TP = take_profit							# take Profit
		self.OT = row['Date'] + ' ' + row['Time']		# open time
		self.CT = None		 							# close time
		self.OP = 0 									# open price
		self.CP = 0										# close price
		self.Profit = 0									# calculated Profit
		self.Instrument = instrument.upper()			# instrument
		self.Closed = False 							# to check if the trade is closed
		# set opening price based on order Type
		if self.Type == 'BUY':
			self.OP = row['AO']
		else:
			self.OP = row['BO']
	

	def asdict(self):
		return {'Type': self.Tpye,
				'Units': self.Units,
				'SL': self.SL,
				'TP': self.TP, 
				'OT': self.OT,
				'CT': self.CT,
				'OP': self.OP,
				'CP': self.CP,
				'Profit': self.Profit,
				'Instrument': self.Instrument,
				'Closed': self.Closed}
	

	def update(self, row):
		if self.Type == 'BUY':
			self.Profit = (row['BO'] - self.OP) * (1 / row['BO']) * self.Units
		elif self.Type == 'SELL':
			self.Profit =  (self.OP - row['AO']) * (1 / row['AO']) * self.Units

		if self.SL != 0 and self.Profit <= -self.SL:	
			self.close(row)
			return True
		if self.TP != 0 and self.Profit >= self.TP:
			self.close(row)
			return True
		return False
	

	def close(self, row):
		self.CT = row['Date'] + ' ' + row['Time']	
		if self.Type == 'BUY':
			self.CP = row['BO']
		elif self.Type == 'SELL':
			self.CP = row['AO']
		self.Closed = True
		