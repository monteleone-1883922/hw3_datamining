
from flask import Flask, render_template, request, redirect, url_for
from k_means_analysis_amazon_products import process_raw_data,apply_feature_engineering
from utils import Representation,load_and_retype_data,load_raw_data
from markupsafe import Markup

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')



@app.route('/visualize_cluster',methods=['post'])
def view_cluster():
    technique = request.form['technique']
    clusters = int(request.form['clusters'])
    cluster = int(request.form['cluster'])
    clean_data = data = load_and_retype_data()
    if technique.find(Representation.RAW.value) != -1:
        data = load_raw_data()
        clustered_data = process_raw_data(data=data,cluster=clusters)
    else:
        pca = technique.find("pca") != -1
        clustered_data = apply_feature_engineering(clean_data,cluster=clusters,pca=pca)
    idxs = [i for i in range(len(clustered_data)) if clustered_data[i] == cluster]
    clustered_products = clean_data.loc[idxs]
    table = clustered_products.to_html()
    table = Markup(table)
    return render_template('visualization.html', table=table, technique=technique, num_clusters=clusters,cluster=cluster)




if __name__ == '__main__':
    app.run(debug=True)


