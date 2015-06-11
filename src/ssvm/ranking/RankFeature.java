package ssvm.ranking;

import java.util.ArrayList;
import java.util.List;

public class RankFeature {

	// list of labels
	private List<Integer> gtLabels = null;
	// list of features
	private List<double[]> features = null;
	// indices for positives samples
	private List<Integer> lp = null;
	// indices for negatives samples
	private List<Integer> ln = null;
	// dimension features
	private int dim;
	
	public RankFeature(List<double[]> features, List<Integer> gtLabels) {
		this.features = features;
		this.gtLabels = gtLabels;
		dim = features.get(0).length;
		update();
	}
	
	public void update() {
		lp = new ArrayList<Integer>();
		ln = new ArrayList<Integer>();
		for(int i=0; i<gtLabels.size(); i++) {
			if(gtLabels.get(i) == 1) {
				lp.add(i);
			}
			else if(gtLabels.get(i) == -1) {
				ln.add(i);
			}
		}
	}
	
	public List<Integer> getGtLabels() {
		return gtLabels;
	}
	public void setGtLabels(List<Integer> gtLabels) {
		this.gtLabels = gtLabels;
	}
	public List<double[]> getFeatures() {
		return features;
	}
	public void setFeatures(List<double[]> features) {
		this.features = features;
	}
	public List<Integer> getLp() {
		return lp;
	}
	public void setLp(List<Integer> lp) {
		this.lp = lp;
	}
	public List<Integer> getLn() {
		return ln;
	}
	public void setLn(List<Integer> ln) {
		this.ln = ln;
	}
	public int getDim() {
		return dim;
	}
	public void setDim(int dim) {
		this.dim = dim;
	}
	
	public List<Integer> getGtRanking() {
		List<Integer> ranking = new ArrayList<Integer>();
		for(int i=0; i<gtLabels.size(); i++) {
			if(gtLabels.get(i) == 1) {
				ranking.add(i);
			}
		}
		for(int i=0; i<gtLabels.size(); i++) {
			if(gtLabels.get(i) == -1) {
				ranking.add(i);
			}
		}
		return ranking;
	}
	
	public String toString() {
		String s = "Ranking Feature \t dim= " + dim + "\n";
		s += "lp= " + lp + "\n";
		s += "ln= " + ln + "\n";
		s += "gt labels= " + gtLabels + "\n";
		s += "rank= " + getGtRanking() + "\n";
		return s;
	}
}
