package latent.variable;

import java.util.ArrayList;
import java.util.List;

public class BagMIL{

	private List<double[]> features = null;
	private List<String> fileFeatures = null;
	private int label;
	private String name;
	
	public BagMIL() {
		features = new ArrayList<double[]>();
		fileFeatures = new ArrayList<String>();
		label = 0;
	}
	
	public BagMIL(BagMIL bag) {
		features = bag.getFeatures();
		label = bag.label;
		name = bag.name;
	}
	
	public void addFeature(double[] f) {
		features.add(f.clone());
	}
	
	public void addFeature(int ind, double[] f) {
		features.add(ind,f.clone());
	}
	
	public double[] getFeature(int ind) {
		return features.get(ind);
	}
	
	public void setFeature(int ind, double[] feature) {
		features.set(ind,feature);
	}

	public int getLabel() {
		return label;
	}

	public void setLabel(int label) {
		this.label = label;
	}

	public void setFeatures(List<double[]> features) {
		this.features = features;
	}
	
	public List<double[]> getFeatures() {
		return features;
	}
	
	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}
	
	public void addFileFeature(String s) {
		fileFeatures.add(s);
	}
	
	public void addFileFeature(int ind, String s) {
		fileFeatures.add(ind,s);
	}
	
	public String getFileFeature(int ind) {
		return fileFeatures.get(ind);
	}
	
	public void removeFileFeature(String s) {
		for(int i=0; i<fileFeatures.size(); i++) {
			if(fileFeatures.get(i).compareTo(s) == 0) {
				fileFeatures.remove(i);
				features.remove(i);
			}
		}
	}

	public String toString() {
		if(features != null) {
			return "name: " + name + "\tlabel: " + label + "\tfeatures: " + features.size() + " x " + features.get(0).length;
		}
		else {
			return "name: " + name + "\tlabel: " + label + "\tfeatures: " + features.size();
		}
	}

}
