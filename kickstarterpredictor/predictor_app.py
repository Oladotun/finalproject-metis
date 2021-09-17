# minimal example from:
# http://flask.pocoo.org/docs/quickstart/

from flask import Flask, render_template, request
# from flask_bootstrap import Bootstrap
# import arrow
import pickle
import numpy as np
# from sklearn.externals import joblib
# from keras.models import load_model
import pandas as pd
from tensorflow.python.keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)  # create instance of Flask class

# bootstrap = Bootstrap(app)
with open("lrResultNonScale_alldata.pkl", "rb") as f:
    logistic_result_model = pickle.load(f)
with open("logisticNeuralNetMaking_alldata.pkl", "rb") as f:
    lr_model = pickle.load(f)
with open("pipeline_alldata.pkl", "rb") as f:
    pipeline_model = pickle.load(f)
name_neural_net_model = load_model('nameNeuralNet_alldata.h5')
blurb_neural_net_model = load_model('blurbNeuralNet_alldata.h5')

category = pd.read_csv('category_unique.csv')
city_state = pd.read_csv('city_state.csv')

standardScaleData = pd.read_csv('standardScaleData.csv')

category_values = np.sort(category.values.flatten())
city_state_values = np.sort(city_state.values.flatten())



original_neural_probs = 0
original_blurb_probs = 0
original_pred_probs = 0
original_logistic_probs = 0
goal_global = 0
duration_global = 0
proj_name = ""
proj_desc  = ""

def getwordgoing(word):
    maxlen = 100
    print("word is: ", word)
    word_array = [[eachWord] for eachWord in word.split(" ")]
    print(word_array)
    tokenizer = Tokenizer(num_words = len(word_array), filters="""!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',""")
    
    print("token word",tokenizer.fit_on_texts([word]))

    word_index = tokenizer.word_index

    print("word index made: ",word_index)
    X_blurb = tokenizer.texts_to_sequences([word])
    print("word to sequence is:")
    print(X_blurb)
    X_blurb = pad_sequences(X_blurb, maxlen=maxlen)

    return X_blurb


def standardScaleGoals(dataFrameValues):
	ss = StandardScaler()
	scaled_df = ss.fit_transform(dataFrameValues)

	# data_frame = pd.DataFrame(scaled_df, columns=['goal','duration_for_days'])

	# dataFrameValues['goal_scaled'] = data_frame['goal']
	# dataFrameValues['duration_for_days_scaled'] = data_frame['duration_for_days']
	# print(scaled_df)
	return ss,scaled_df

def standardScale(predictNameProb,predictblurbProb,prediction_test_proba):
	# print("work", list(prediction_test_proba))
	if len(prediction_test_proba) == 0:
		return 

	print("predict test proba ", prediction_test_proba[1])
	print("predict name proba ", predictNameProb)
	print("predict neural proba ", predictblurbProb)
	probToResult = pd.DataFrame(
	{'predict_name': predictNameProb,
	 'predict_blurb': predictblurbProb,
	 'logistic_prediction' : prediction_test_proba[1]
	})

	# cols = ['predict_name', 'predict_blurb','logistic_prediction']
	# subset_df = probToResult[cols]
	# #Standard Scaler

	# print("going to standard scale")
	# ss = StandardScaler()
	# scaled_df_prob = ss.fit_transform(subset_df)
	# scaled_df_prob = pd.DataFrame(scaled_df_prob, columns=cols)

	return probToResult

pred_ss = standardScaleGoals(standardScaleData)[0]
scaled_dataframe = standardScaleGoals(standardScaleData)[1]

# print(pred_ss.head(5))
# url form http://127.0.0.1:5000/action_page.php?sepal+length+%28cm%29=0&sepal+width+%28cm%29=0&petal+length+%28cm%29=0&petal+width+%28cm%29=0
@app.route("/predict", methods=["POST", "GET"])
def predict():
	x_input = []
	name_input = []
	blurb_input = []
	feature_names_map = {}
	feature_names_array = ['goal','duration_for_days'] 
	neural_probs = []
	blurb_probs = []
	pred_probs = []
	num = 0
	numy = 0
	numz = 0
	numr = 0

	global original_neural_probs 
	global original_blurb_probs 
	global original_pred_probs 
	global original_logistic_probs
	global proj_name
	global proj_desc 
	global goal_global
	global duration_global

	# original_neural_probs = 0
	# original_blurb_probs = 0
	# original_pred_probs = 0
	# original_logistic_probs = 0


	for i in range(len(lr_model.feature_names)):
		feature_names_map[lr_model.feature_names[i]] = 0

	categories_val = request.args.get("categories", "")
	goal = float(request.args.get("goal", "0"))
	# staff = float(request.args.get("staff_pick", "0"))
	duration = float(request.args.get("duration_for_days", "0"))
	# day_of_week= request.args.get("day", "")
	# time_of_day = request.args.get("time", "")

	# located_state = request.args.get("city_state", "")

	feature_names_map["goal"] = goal
	# feature_names_map["staff_pick"] = staff
	feature_names_map["duration_for_days"] = duration
	# or staff != 0
	# or day_of_week != "" 

	if goal != 0  or duration != 0 or categories_val != "":
		pd_goal_duration = pd.DataFrame({
		"goal": [goal],
		"duration_for_days": [duration]
		})

		duration_global = duration
		goal_global = goal

		
		# print("Prediction result yes")
		# print(duration_result)
		# print(goal_result)
		

		# print("Test probability result")
		# print(pd_goal_duration.values)
		# print(pred_ss.transform(pd_goal_duration.values))

		feature_names_map["goal"] = pred_ss.transform(pd_goal_duration.values)[:,0]

		# for 

		feature_names_map["duration_for_days"] = pred_ss.transform(pd_goal_duration.values)[:,1]


		for i in feature_names_map.keys():
			if i == categories_val:
				# print("updating categories")
				# print("categories is", categories_val)
				feature_names_map[categories_val] = 1
			# if i == day_of_week:
			# 	feature_names_map[day_of_week] = 1
			# if i == time_of_day:
			# 	feature_names_map[time_of_day] = 1
			# if i == located_state:
			# 	feature_names_map[located_state] = 1

		feature_names_map_values = list(feature_names_map.values())

		# print(feature_names_map)

		# print("length of keys",len(feature_names_map.keys()))
		pred_probs = list(lr_model.predict_proba([feature_names_map_values]).flat)
		pred_regular = list(lr_model.predict([feature_names_map_values]).flat)
		# print("pred prob", pred_probs)
		# print("pred regular ", pred_regular[0])
		numr = pred_probs[1] * 100

	name_value = request.args.get("name", "")
	if name_value != "":
		neural_probs = list(name_neural_net_model.predict(getwordgoing(name_value)).flat)
	
	proj_name = name_value

	blurb_value = request.args.get("blurb", "")
	if blurb_value != "":
		blurb_probs = list(blurb_neural_net_model.predict(getwordgoing(blurb_value)).flat)

	proj_desc = blurb_value
	# print("Enter project description")
	# print(proj_desc)
	if len(blurb_probs) > 0 or len(neural_probs) > 0 or len(pred_probs) > 0:
		result = standardScale(neural_probs,blurb_probs,pred_probs)
		# print("Data result")
		# print(result)
		result_probs = 0

		print(neural_probs)
		print(blurb_probs)
		numy = neural_probs[0]* 100
		numz = blurb_probs[0] * 100

		if result is not None and len(result.index)>0:
			result_probs = pipeline_model.predict_proba(result[['predict_name','predict_blurb','logistic_prediction']])
			num = result_probs[:,1] * 100
			# print("result probs is ",num)
			# print("result probs trying is ",result_probs)
			num = num[0]	
	# print("num is ",num)

	x = '{:.2f}'.format(num)
	y = '{:.2f}'.format(numy)
	z = '{:.2f}'.format(numz)
	r = '{:.2f}'.format(numr)
	# print("enter pred result")

	print("original pred is x: ", x)
	print("original logistic is y: ", y)
	print("original blurb is z : ", z)
	print("evaluate statement result: ", original_pred_probs == 0 )
	print("evaluate statement result: ", original_logistic_probs == 0 )
	print("evaluate statement result: ",  original_blurb_probs == 0 )
	print("evaluate statement result: ",  original_neural_probs == 0)


	print("Original pred ", original_pred_probs )
	print("Original logistic: ", original_logistic_probs)
	print("Original blurb ",  original_blurb_probs )
	print("Original neural nets ",  original_neural_probs )

	

	if float(original_pred_probs) == 0.0 :
		print("storing number")
		original_pred_probs = x
	if float(original_logistic_probs) == 0.0:
		original_logistic_probs = r
	if float(original_blurb_probs) == 0.0 :
		original_blurb_probs = z
	if float(original_neural_probs) == 0.0:
		original_neural_probs = y


	# print("evaluate statement result: ",  int(x) == 0.0)
	# print("evaluate statement result: ",  r == 0.0)
	# print("evaluate statement result: ",  z == 0.0)
	# print("evaluate statement result: ",  y == 0.0)

	if num == 0:
		original_pred_probs = x
	if numr == 0:
		original_logistic_probs = r
	if numz == 0:
		original_blurb_probs = z
	if numy == 0:
		original_neural_probs = y



	# print("original pred is: ", original_pred_probs)
	# print("original logistic is: ", original_logistic_probs)
	# print("original blurb is: ", original_blurb_probs)
	# print("original neural is: ", original_neural_probs)

	# print("x number is", x)
	# print(neural_probs)
	# print(blurb_probs)
	# print(result_probs)
	# print(pred_probs)

	# print(x)
	# print(y)
	# print(z)
	# print(r)

	# print("Enter project description again")
	# print(proj_desc)

	
	return render_template('predictorcontact.html',
	feature_names=feature_names_array,
	x_input=[ goal_global,duration_global],
	name_input=['Goal Amount','Duration'],
	resultpred = x,
	# states = city_state_values,
	# ['10009 NY', '19901 DE', '20165 VA', '32573 FL', '59602 MT', '93041 CA', '99352 WA', 'Aalst Oost-Vlaanderen', 'Accra Greater Accra', 'Adelaide SA', 'Adrian MI', 'Ahwahnee CA', 'Aiken SC', 'Akron OH', 'Alabaster AL', 'Alameda CA', 'Albany CA', 'Albany NY', 'Albuquerque NM', 'Alexandria VA', 'Allentown PA', 'Alpharetta GA', 'Amarillo TX', 'Ames IA', 'Amesbury MA', 'Amherst MA', 'Amman Amman', 'Anacortes WA', 'Anaheim CA', 'Anchorage AK', 'Andalusia AL', 'Andice TX', 'Ankeny IA', 'Ann Arbor MI', 'Annapolis MD', 'Antioch CA', 'Apalachin NY', 'Apex NC', 'Apple Valley MN', 'Aptos CA', 'Arcata CA', 'Arlington MA', 'Arlington TX', 'Asheville NC', 'Ashland OR', 'Aspen CO', 'Assisi Umbria', 'Astoria US', 'Atascadero CA', 'Athens GA', 'Athens OH', 'Atlanta GA', 'Atlantic City NJ', 'Auburn NY', 'Auburn WA', 'Augusta GA', 'Aurora IL', 'Austin TX', 'Avarua Rarotonga', 'Avon Park US', 'Azusa CA', 'Bakersfield CA', 'Baltimore MD', 'Banff AB', 'Barberton OH', 'Bariloche Rio Negro', 'Baton Rouge LA', 'Bay City MI', 'Beacon NY', 'Beardstown IL', 'Beaufort SC', 'Beaumont TX', 'Bedford NH', 'Beggs OK', 'Belgrade Beograd', 'Bellingham WA', 'Bend OR', 'Benzonia MI', 'Berea KY', 'Berkeley CA', 'Berlin Berlin', 'Berlin NJ', 'Berwyn IL', 'Bethesda MD', 'Bethpage TN', 'Big Bear Lake CA', 'Binghamton NY', 'Birmingham AL', 'Black Rock City NV', 'Bloomington IN', 'Blue River CO', 'Bluffton SC', 'Boca Raton FL', 'Boise ID', 'Bondville VT', 'Boone NC', 'Boscawen NH', 'Boston MA', 'Boulder CO', 'Bow WA', 'Bowie MD', 'Bowling Green OH', 'Bowman SC', 'Boyden IA', 'Boyne City MI', 'Bozeman MT', 'Bradenton FL', 'Branchland WV', 'Brandon FL', 'Brattleboro VT', 'Bremerton WA', 'Brevard NC', 'Bridgewater VA', 'Brockport NY', 'Broken Arrow OK', 'Bronx NY', 'Brooklyn NY', 'Broomfield CO', 'Brownsville TX', 'Buckeye AZ', 'Buckhannon WV', 'Buffalo NY', 'Burbank CA', 'Bushwick US', "Cabris Provence-Alpes-Cote d'Azur", 'Caesarea Haifa', 'Cairo Cairo', 'Camarillo CA', 'Cambridge MA', 'Cambridge OH', 'Camden NJ', 'Campbell CA', 'Canton OH', 'Carbondale IL', 'Carlsbad CA', 'Carmel NY', 'Carneys Point NJ', 'Carrollton GA', 'Carson City NV', 'Cary NC', 'Casey IL', 'Castle Rock CO', 'Cathedral City CA', 'Cedar Rapids IA', 'Centreville VA', 'Cerritos CA', 'Chapel Hill NC', 'Charles Town WV', 'Charleston SC', 'Charleston WV', 'Charlotte NC', 'Charlottesville VA', 'Chateaugay NY', 'Chattanooga TN', 'Chesapeake VA', 'Chesterfield MI', 'Chicago IL', 'Chico CA', 'Chillicothe OH', 'Chimaltenango Chimaltenango', 'Chino Hills CA', 'Chowchilla CA', 'Chula Vista CA', 'Chuluota FL', 'Cincinnati OH', 'Civita Castellana Lazio', 'Claremore OK', 'Clarion IA', 'Clarksville TN', 'Clayton CA', 'Cleburne TX', 'Cleveland OH', 'Clovis NM', 'Coachella CA', 'Cochabamba Cochabamba', 'Coimbatore Tamil Nadu', 'Colchester CT', 'Colchester VT', 'Coldwater MS', 'College Station TX', 'Cologne MN', 'Colombia Huila', 'Colorado Springs CO', 'Columbia MD', 'Columbia MO', 'Columbia SC', 'Columbus GA', 'Columbus OH', 'Concord NC', 'Concord NH', 'Conway AR', 'Coos Bay OR', 'Coral Springs FL', 'Corbett US', 'Cordova IL', 'Corning NY', 'Corona CA', 'Corpus Christi TX', 'Corvallis OR', 'Costa Mesa CA', 'Cotonou Littoral', 'Covington GA', 'Covington LA', 'Crestview Hills KY', 'Crestwood KY', 'Crystal Lake IL', 'Cupertino CA', 'Dallas TX', 'Dalton GA', 'Dana Point CA', 'Dar es Salaam Dar es Salaam', 'Dartmouth MA', 'Davenport IA', 'Davis CA', 'Dayton OH', 'Daytona Beach FL', 'De Kalb IL', 'Dearborn MI', 'Decatur IN', 'Delaware OH', 'Deltona FL', 'Denton TX', 'Denver CO', 'Des Moines IA', 'Destin US', 'Detroit MI', 'Dingmans Ferry PA', 'Dothan AL', 'Douglas GA', 'Dover PA', 'Dublin Dublin', 'Dubuque IA', 'Duluth MN', 'Dumont NJ', 'Dunedin Otago', 'Durham NC', 'Dusseldorf North Rhine-Westphalia', 'Eagle Point OR', 'East Brunswick NJ', 'East Durham NY', 'East Lansing MI', 'East Moline IL', 'East Nashville US', 'East Saint Cloud US', 'East Wakefield NH', 'Easthampton MA', 'Edgewood NM', 'Edinburgh Scotland', 'Edmond OK', 'Edmonton AB', 'Efland NC', 'El Cajon CA', 'El Cerrito CA', 'El Paso TX', 'Elk Grove CA', 'Elkridge MD', 'Emporia KS', 'Ephrata PA', 'Erie PA', 'Eugene OR', 'Euless TX', 'Eureka CA', 'Eustis FL', 'Exmore VA', 'Fairbanks AK', 'Fairfax VA', 'Fairfield IA', 'Farmingdale US', 'Fayetteville AR', 'Fayetteville NC', 'Fenton MI', 'Fernandina Beach FL', 'Fillmore IN', 'Firestone CO', 'Flagstaff AZ', 'Flint MI', 'Florence SC', 'Florence Tuscany', 'Flower Mound TX', 'Fontana CA', 'Forest Park GA', 'Forney TX', 'Fort Bliss US', 'Fort Collins CO', 'Fort Davis TX', 'Fort Lauderdale FL', 'Fort Mill SC', 'Fort Morgan CO', 'Fort Wayne IN', 'Fort Worth TX', 'Fortaleza Santa Catarina', 'Fostoria OH', 'Fox Lake IL', 'Frankfort KY', 'Franklin NH', 'Franklin TN', 'Franklin WI', 'Fredericksburg VA', 'Freehold NJ', 'Fremont CA', 'Fresno CA', 'Frisco TX', 'Fullerton CA', 'Fuquay-Varina NC', 'Fussa-shi Tokyo Prefecture', 'Gainesville FL', 'Garden Grove CA', 'Gardena CA', 'Geneva Canton of Geneva', 'Georgetown KY', 'Gig Harbor WA', 'Gilbert AZ', 'Glastonbury CT', 'Glen Head NY', 'Glendale AZ', 'Glendale CA', 'Glennie MI', 'Glens Falls NY', 'Glyndon MN', 'Graham WA', 'Grand Haven MI', 'Grand Rapids MI', 'Green Bay WI', 'Greencastle PA', 'Greendale WI', 'Greensboro NC', 'Greenville NY', 'Greenville SC', 'Greenwich CT', 'Grosse Pointe Woods MI', 'Gurnee IL', 'Gwinn MI', 'Hailey ID', 'Haleiwa HI', 'Hamden CT', 'Hammond LA', 'Harlem US', 'Harlingen TX', 'Harrisburg PA', 'Hartford CT', 'Hartselle AL', 'Hastings Hudson NY', 'Hawthorne CA', 'Hawthorne NJ', 'Hayesville NC', 'Haynesville LA', 'Hayward CA', 'Helsinki Uusimaa', 'Hempstead NY', 'Hermosa Beach CA', 'Herndon VA', 'Hesperia CA', 'High Point NC', 'Highlands Ranch CO', 'Hillsboro OR', 'Hilo HI', 'Hilton Head Island SC', 'Hinsdale IL', 'Holland MI', 'Hollywood FL', 'Hollywood US', 'Hong Kong Hong Kong Island', 'Honokaa HI', 'Honolulu HI', 'Hot Springs AR', 'Houston TX', 'Howell MI', 'Hudson NH', 'Hughson CA', 'Huntington Beach CA', 'Huntington NY', 'Huntley IL', 'Huntsville AL', 'Hurst TX', 'Idaho Falls ID', 'Imlay NV', 'Indian Trail NC', 'Indianapolis IN', 'Indonesia West Java', 'Inglewood CA', 'Iowa City IA', 'Irvine CA', 'Irving TX', 'Istanbul Istanbul', 'Ithaca NY', 'Itta Bena MS', 'Jackson MI', 'Jackson MS', 'Jackson TN', 'Jacksonville FL', 'Jefferson City MO', 'Jenks OK', 'Jersey City NJ', 'Jesup IA', 'Jinja Jinja', 'Johnson City TN', 'Johnson VT', 'Johnstown NY', 'Joshua Tree CA', 'Juneau AK', 'Kailua Kona HI', 'Kansas City KS', 'Kansas City MO', 'Kecskemét Kecskemét', 'Keizer OR', 'Kennesaw GA', 'Kennewick WA', 'Kenosha WI', 'Kent NY', 'Kent OH', 'Kharkiv Kharkiv Oblast', 'Kiev Kiev City Municipality', 'Kigali Kigali Province', 'Kigoma Kigoma', 'Kingsland GA', 'Kingsport TN', 'Kingston NY', 'Kissimmee FL', 'Klamath Falls OR', 'Knoxville TN', 'Kosciusko MS', 'Kronenwetter WI', 'La Mirada CA', 'La Porte IN', 'La Virginia Risaralda', 'Lafayette CO', 'Lafayette IN', 'Lafayette LA', 'Lake Havasu City AZ', 'Lake Itasca MN', 'Lake Stevens WA', 'Lake Worth FL', 'Lakeland FL', 'Lakewood CO', 'Lancaster PA', 'Lansing MI', 'Laramie WY', 'Larkspur CA', 'Las Vegas NM', 'Las Vegas NV', 'Latrobe PA', 'Lawrence KS', 'Lawrenceville GA', 'League City TX', 'Leavenworth KS', 'Lebanon IN', 'Lebanon OR', 'Lehigh Valley PA', 'Lewisville TX', 'Lexington KY', 'Lihue HI', 'Linden US', 'Lisle IL', 'Lithonia GA', 'Lititz PA', 'Little Rock AR', 'Littleton NH', 'Lockport NY', 'Logan UT', 'Lombard IL', 'London England', 'Long Beach CA', 'Long Beach MS', 'Long Island NY', 'Longview WA', 'Longyearbyen Svalbard', 'Lorton VA', 'Los Angeles CA', 'Los Molinos CA', 'Louisville KY', 'Lowell MA', 'Lower East Side US', 'Lubbock TX', 'Lynbrook NY', 'Lynchburg VA', 'Macon GA', 'Madison WI', 'Makati City National Capital Region', 'Makawao HI', 'Malibu CA', 'Malvern PA', 'Mamaroneck NY', 'Manchester England', 'Manhattan KS', 'Manhattan NY', 'Manistee MI', 'Mankato MN', 'Manteo NC', 'Mapleton UT', 'Marcos Juárez Cordoba', 'Marfa TX', 'Marlinton WV', 'Marquette MI', 'Marrakesh مراكش ـ تانسيفت ـ الحوز', 'Marshall NC', 'Marysville OH', 'Marysville WA', 'Mcallen TX', 'Mcminnville OR', 'Meadville PA', 'Medellin Antioquia', 'Medford OR', 'Media PA', 'Melbourne VIC', 'Memphis TN', 'Mendota IL', 'Menlo Park CA', 'Menomonie WI', 'Merced CA', 'Meridian ID', 'Metairie LA', 'Miami Beach FL', 'Miami FL', 'Middletown CT', 'Midland MI', 'Midland TX', 'Midlothian VA', 'Milwaukee WI', 'Minneapolis MN', 'Minocqua WI', 'Minot ND', 'Minsk Minsk', 'Miramar FL', 'Mishawaka IN', 'Mission Beach US', 'Mobile AL', 'Modesto CA', 'Monroe WA', 'Monrovia CA', 'Monsey NY', 'Montclair NJ', 'Montebello CA', 'Montgomery AL', 'Montreal QC', 'Montrose CO', 'Monument Valley UT', 'Moreno Valley CA', 'Morristown TN', 'Moscow ID', 'Moscow Moscow Federal City', 'Mount Laurel NJ', 'Mountain View CA', 'Murrells Inlet SC', 'NE Portland US', 'Nairobi Nairobi Area', 'Nampa ID', 'Naples FL', 'Nashua NH', 'Nashville TN', 'Natick MA', 'New Bern NC', 'New Britain CT', 'New Brunswick NJ', 'New Haven CT', 'New Milford CT', 'New Orleans LA', 'New Paltz NY', 'New Smyrna Beach FL', 'New York NY', 'Newark NJ', 'Newaygo MI', 'Newbury Park US', 'Newport Beach CA', 'Newport KY', 'Newport News VA', 'Niagara Falls NY', 'Nimes Languedoc-Roussillon', 'Norcross GA', 'Norfolk VA', 'Norman OK', 'North Adams MA', 'North Atlanta GA', 'North Bennington VT', 'North Haledon NJ', 'North Hollywood US', 'North Plainfield NJ', 'North Pole AK', 'North Ridgeville OH', 'Northampton MA', 'Norwalk CT', 'Nowhere OK', 'Oak Grove US', 'Oak Park IL', 'Oakland CA', 'Oakland ME', 'Oakton VA', 'Ocala FL', 'Ocean City NJ', 'Oceanside CA', 'Ojai CA', 'Oklahoma City OK', 'Okmulgee OK', 'Old Town ME', 'Olean NY', 'Olympia WA', 'Omaha NE', 'Orange CT', 'Oregon City OR', 'Orem UT', 'Orlando FL', 'Osaka-shi Osaka Prefecture', 'Oshkosh WI', 'Oslo Oslo Fylke', 'Pahokee FL', 'Palestine TX', 'Palo Alto CA', 'Palos Verdes Estates CA', 'Panacea FL', 'Panama City FL', 'Paris Ile-de-France', 'Park City UT', 'Parowan UT', 'Parrish FL', 'Parsippany NJ', 'Pasadena CA', 'Paseo Artists District US', 'Peabody MA', 'Pekin IN', 'Pensacola FL', 'Peoria AZ', 'Peoria IL', 'Pepperell MA', 'Perry NY', 'Peru IN', 'Petaluma CA', 'Philadelphia PA', 'Phoenix AZ', 'Pittsburgh PA', 'Placerville CA', 'Plano TX', 'Plymouth MA', 'Pocatello ID', 'Pomona CA', 'Pompano Beach FL', 'Poolesville MD', 'Port Huron MI', 'Portage WI', 'Portland ME', 'Portland OR', 'Portsmouth OH', 'Post Falls ID', 'Pottstown PA', 'Poughkeepsie NY', 'Poulsbo WA', 'Prairieville LA', 'Princeton NJ', 'Princeville HI', 'Prosser WA', 'Providence RI', 'Provo UT', 'Pueblo CO', 'Pullman WA', 'Punta Gorda FL', 'Puyallup WA', 'Queens NY', 'Raeford NC', 'Rahway NJ', 'Raleigh NC', 'Rancho Cucamonga CA', 'Randolph NJ', 'Reading PA', 'Redlands CA', 'Redwood NY', 'Reno NV', 'Reston VA', 'Rexburg ID', 'Reykjavik Reykjavik', 'Rice Lake WI', 'Richmond CA', 'Richmond VA', 'Ridgefield CT', 'Rigby ID', 'Ringgold GA', 'Rio de Janeiro Rio de Janeiro', 'River Falls WI', 'Riverhead NY', 'Riverside CA', 'Rochester MN', 'Rochester NY', 'Rock Island IL', 'Rockaway NJ', 'Rockford IL', 'Rockport ME', 'Rockville MD', 'Rohnert Park CA', 'Roseboro NC', 'Roseville CA', 'Rowley MA', 'Ruidoso NM', 'Rushville MO', 'S. Andres Peten', 'Sacramento CA', 'Saint James NY', 'Salem MA', 'Salem NH', 'Salem OR', 'Salinas CA', 'Salt Lake City UT', 'Sammamish WA', 'San Angelo TX', 'San Antonio TX', 'San Antonio US', 'San Bernardino CA', 'San Diego CA', 'San Francisco CA', 'San Jose CA', 'San Luis Obispo CA', 'San Marcos CA', 'San Marcos TX', 'San Marcos la Laguna Solola', 'San Rafael CA', 'San Tan Valley AZ', 'Sandpoint ID', 'Sandy UT', 'Sanford FL', 'Santa Ana CA', 'Santa Barbara CA', 'Santa Clara CA', 'Santa Cruz CA', 'Santa Fe NM', 'Santa Monica CA', 'Santa Rosa CA', 'Santo Domingo Distrito Nacional', 'Sao Paulo Sao Paulo', 'Sarasota FL', 'Savannah GA', 'Scottsbluff NE', 'Scottsdale AZ', 'Seattle WA', 'Seven Points TX', 'Shafer MN', 'Shawnee OK', 'Shelton CT', 'Sheridan AR', 'Sherwood AR', 'Shreveport LA', 'Silver Spring MD', 'Singapore Central Singapore', 'Sioux City IA', 'Sioux Falls SD', 'Sisters OR', 'Smyrna TN', 'Sneads Ferry NC', 'Snellville GA', 'Snoqualmie WA', 'Snowflake AZ', 'Socorro NM', 'Somerville MA', 'South Florida FL', 'South Jordan UT', 'South Orange NJ', 'South Pasadena CA', 'Southfield MI', 'Sparks NV', 'Spokane WA', 'Spooner WI', 'Spring Lake NJ', 'Springfield MA', 'Springfield MO', 'Springfield OH', 'Springville UT', 'Spur TX', 'St. Augustine FL', 'St. Charles IL', 'St. Charles MO', 'St. George UT', 'St. John St. John', "St. John's MB", 'St. Louis MO', 'St. Paul MN', 'St. Pete Beach FL', 'St. Petersburg FL', 'Stamford CT', 'Stanchfield MN', 'State College PA', 'Staten Island NY', 'Staunton VA', 'Sterling IL', 'Sterling VA', 'Stevens Point WI', 'Stillwater OK', 'Stockbridge MA', 'Stoughton WI', 'Stow OH', 'Stratford NJ', 'Studio City US', 'Suffolk VA', 'Sunnyvale CA', 'Sussex NJ', 'Sutherlin OR', 'Suwanee GA', 'Swansea MA', 'Syracuse NY', 'Szczecin West Pomeranian', 'Tacoma WA', 'Tallahassee FL', 'Tampa FL', 'Tangent OR', 'Tarboro NC', 'Tavares FL', 'Taylors SC', 'Tbilisi T´bilisi', 'Tecopa CA', 'Tempe AZ', 'The Hague South Holland', 'The Plains VA', 'Thomasville GA', 'Thousand Oaks US', 'Tinley Park IL', 'Tiruvannamalai Tamil Nadu', 'Tokyo Tokyo Prefecture', 'Toledo OH', 'Tonawanda NY', 'Topeka KS', 'Toronto ON', 'Towson MD', 'Trabuco Canyon CA', 'Traverse City MI', 'Trenton NJ', 'Troy NY', 'Truckee CA', 'Tucson AZ', 'Tulsa OK', 'Tulum Quintana Roo', 'Tuscaloosa AL', 'Twin Falls ID', 'Ukiah CA', 'Ulan Bator Ulaanbaatar', 'Unionville NY', 'University Park PA', 'Upland CA', 'Utica NY', 'Vail CO', 'Valdosta GA', 'Vallejo CA', 'Valparaiso IN', 'Van Dyne WI', 'Vancouver BC', 'Vancouver WA', 'Vandalia OH', 'Venezuela Zulia', 'Venice US', 'Venice Veneto', 'Ventura CA', 'Vernon CT', 'Vienna Vienna', 'Virginia Beach VA', 'Visalia CA', 'Waco TX', 'Waldorf MD', 'Warren PA', 'Warrensburg MO', 'Warwick RI', 'Washington DC', 'Watsonville CA', 'Waukesha WI', 'Wayne NJ', 'Wellington FL', 'Wenatchee WA', 'Wesley Chapel FL', 'West Chester PA', 'West Hartford CT', 'West Hollywood CA', 'West Lake Wales US', 'West Nashville US', 'West Palm Beach FL', 'West Sacramento CA', 'West des Moines IA', 'Westbury NY', 'Westchester US', 'Westfield MA', 'Westlake Village CA', 'Westminster CO', 'Westminster MD', 'Westport CT', 'White Bear Lake MN', 'White Sulphur Springs WV', 'Wichita KS', 'Wilmington DE', 'Wilmington NC', 'Winchester VA', 'Windham ME', 'Winston-Salem NC', 'Winter Park FL', 'Woodbridge VA', 'Woodbury CT', 'Woodbury NJ', 'Woodinville WA', 'Woodstock GA', 'Worcester MA', 'Yoakum TX', 'York PA', 'Yosemite Village CA', 'Ypsilanti MI', 'Yucaipa CA', 'Zolfo Springs FL'],
	# day = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
	categories = category_values,
	neuralpred = y,
	blurbpred = z,
	logisticpred = r,

	original_pred_probs = original_pred_probs,
	original_blurb_probs = original_blurb_probs,
	original_neural_probs = original_neural_probs,
	original_logistic_probs  = original_logistic_probs ,

	name_message = proj_name,
	blurb_message = proj_desc,
	goal_pred = goal_global,
	duration_global= duration_global )

	

	# time=['Morning', 'Afternoon', 'Night']
	# ['Academic publishing', 'Accessories fashion', 'Action film & video', 'Animals photography', 'Animation film & video', 'Anthologies comics', 'Anthologies publishing', 'Apparel fashion', 'Apps technology', 'Architecture design', 'Art Books publishing', 'Art art', 'Audio journalism', 'Bacon food', 'Blues music', 'Calendars publishing', 'Camera Equipment technology', 'Candles crafts', 'Ceramics art', "Children's Books publishing", 'Childrenswear fashion', 'Civic Design design', 'Classical Music music', 'Comedy film & video', 'Comedy music', 'Comedy theater', 'Comic Books comics', 'Community Gardens food', 'Conceptual Art art', 'Cookbooks food', 'Country & Folk music', 'Couture fashion', 'Crafts crafts', 'Crochet crafts', 'DIY Electronics technology', 'DIY crafts', 'Dance dance', 'Digital Art art', 'Documentary film & video', 'Drama film & video', 'Drinks food', 'Electronic Music music', 'Events comics', 'Experimental film & video', 'Fabrication Tools technology', 'Faith music', 'Family film & video', 'Fantasy film & video', "Farmer's Markets food", 'Farms food', 'Fashion fashion', 'Festivals film & video', 'Festivals theater', 'Fiction publishing', 'Fine Art photography', 'Flight technology', 'Food Trucks food', 'Food food', 'Footwear fashion', 'Gadgets technology', 'Glass crafts', 'Graphic Design design', 'Graphic Novels comics', 'Hardware technology', 'Hip-Hop music', 'Illustration art', 'Immersive theater', 'Indie Rock music', 'Installations art', 'Interactive Design design', 'Jazz music', 'Jewelry fashion', 'Journalism journalism', 'Knitting crafts', 'Latin music', 'Letterpress publishing', 'Literary Spaces publishing', 'Live Games games', 'Makerspaces technology', 'Mixed Media art', 'Mobile Games games', 'Movie Theaters film & video', 'Music Videos film & video', 'Music music', 'Musical theater', 'Narrative Film film & video', 'Nature photography', 'Nonfiction publishing', 'Painting art', 'People photography', 'Performance Art art', 'Performances dance', 'Periodicals publishing', 'Pet Fashion fashion', 'Photo journalism', 'Photobooks photography', 'Photography photography', 'Playing Cards games', 'Plays theater', 'Poetry publishing', 'Pop music', 'Pottery crafts', 'Print journalism', 'Printing crafts', 'Product Design design', 'Public Art art', 'Publishing publishing', 'Quilts crafts', 'R&B music', 'Radio & Podcasts publishing', 'Ready-to-wear fashion', 'Restaurants food', 'Robots technology', 'Rock music', 'Romance film & video', 'Science Fiction film & video', 'Sculpture art', 'Shorts film & video', 'Small Batch food', 'Software technology', 'Sound technology', 'Space Exploration technology', 'Spaces dance', 'Spaces food', 'Spaces theater', 'Stationery crafts', 'Tabletop Games games', 'Technology technology', 'Theater theater', 'Thrillers film & video', 'Translations publishing', 'Typography design', 'Vegan food', 'Video Games games', 'Video journalism', 'Wearables technology', 'Web journalism', 'Web technology', 'Webcomics comics', 'Webseries film & video', 'Woodworking crafts', 'Workshops dance', 'Young Adult publishing', 'Zines publishing']
	# )


if __name__ == '__main__':
    app.run(debug=True)
