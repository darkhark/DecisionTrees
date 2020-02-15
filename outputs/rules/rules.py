#This rule was reconstructed from outputs/rules/rules.json
def findDecision(obj):
	if obj[1] == 'Out':
		if obj[2] == 'NBC':
			if obj[0] == 'Home':
				return 'WIN'
			elif obj[0] == 'Away':
				return 'WIN'
		elif obj[2] == 'ABC':
			if obj[0] == 'Away':
				return 'WIN'
		elif obj[2] == 'ESPN':
			return 'WIN'
		elif obj[2] == 'CBS':
			return 'LOSS'
	elif obj[1] == 'In':
		if obj[0] == 'Home':
			if obj[2] == 'NBC':
				return 'WIN'
		elif obj[0] == 'Away':
			return 'LOSS'
