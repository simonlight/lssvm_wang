package graph.variable;

import java.util.ArrayList;
import java.util.List;

import util.VectorOp;

public class NormProduct {
	
	public static void inference(int[][] adjMatrix, double[][] logLocalPotentials, int[][] cPair, double[][] cCond, double[] cLocal) {
		
		int totalVertices = adjMatrix.length;
		//veryfing inputs
		
		
		// algorithm start
		double[][] cHatCond = VectorOp.add(cPair, cCond);
		double[] cHatLocal = new double[totalVertices];
		for(int i=0; i<cHatLocal.length; i++) {
			cHatLocal[i] += cLocal[i];
			for(int j=0; j<cPair[i].length; j++) {
				cHatLocal[i] += cPair[i][j];
			}
		}
		
		int[] totalValues = new int[totalVertices];
		double[][] logLocalBeliefs = new double[totalVertices][];
		List<List<Integer>> neighbors = new ArrayList<List<Integer>>();
		for(int i=0; i<totalVertices; i++) {
			totalValues[i] = logLocalPotentials[i].length;
			logLocalBeliefs[i] = new double[totalValues[i]];
			neighbors.add(new ArrayList<Integer>());
			for(int j=0; j<adjMatrix[i].length; j++) {
				if(adjMatrix[i][j] == 1) {
					neighbors.get(i).add(j);
				}
			}
		}
		
		//logMessages_n{i,j}(xi,xj) means log n_{i->(ij)}(xi,xj) in the paper;
		//logMessages_m{j,i}(xi) means log m_{(ij)->i}(xi) in the paper;
		// possibilité d'initialiser et de le passer en parametre
		// initialise message
		double[][][][] logMessagesN = new double[totalVertices][totalVertices][][];
		double[][][] logMessagesM 	= new double[totalVertices][totalVertices][];
		for(int i=0; i<totalVertices; i++) {
			for(int j=0; j<totalVertices; j++) {
				if(adjMatrix[i][j] == 1) {
					logMessagesN[i][j] = new double[totalValues[i]][totalValues[j]];
					logMessagesM[i][j] = new double[totalValues[j]];
				}
			}
		}
		
		// test convexité
		boolean convexFlag = false;
		if((VectorOp.min(cPair) >= 0) && (VectorOp.min(cCond) >= 0) && (VectorOp.min(cLocal) >= 0)) {
			convexFlag = true;
		}
		else {
			
		}
		System.out.println("convexe ? " + convexFlag);
		
		int totalIterations = 100;
		for(int t=0; t<totalIterations; t++) {
			System.out.println("iter " + t);
			
			if(convexFlag) {
				
			}
			else {
				
			}
			
			for(int i=0; i<totalVertices; i++) {
				
				// computing the logMessagesM for a given i, for all the xi in parallel
				for(int j : neighbors.get(i)) {
					
				}
				
				// compute the log of local beliefs up to additive scale
		        logLocalBeliefs[i] = logLocalPotentials[i];
		        for(int j : neighbors.get(i)) {
		        	for(int k=0; k<logLocalBeliefs[i].length; k++) {
		        		logLocalBeliefs[i][k] += logMessagesM[j][i][k];
		        	}
		        }
		        
		        // normalization to the logBeliefs
		        double mean = VectorOp.mean(logLocalBeliefs[i]);
		        for(int k=0; k<logLocalBeliefs[i].length; k++) {
			        logLocalBeliefs[i][k] = logLocalBeliefs[i][k] / cHatLocal[i];
			        logLocalBeliefs[i][k] = logLocalBeliefs[i][k] - mean; 
		        }
			}
			
		}
		
	}
	
	
	public static double getDualValue(double[][][][] logPairPotentials, double[][][] logLocalPotentials,int totalVertices, 
			List<List<Integer>> neighbors, double[][][][] logMessagesN, int[][] cPair, double[][] cCond, double[] cLocal, double epsilon) {

		double dualValue=0;
		
		for(int i=0; i<totalVertices; i++) {
			for(int j : neighbors.get(i)) {
				if(j>i) {
					continue;
				}
				double[] v = null;
				for(int k=0; k<logMessagesN[i][j].length; k++) {
					
				}
				dualValue -= logNorm(v, 1/(epsilon*cPair[i][j]));
			}
		}
		
		
		return dualValue;
	}
	
	private static double logNorm(double[] v, double p) {
		double m = VectorOp.max(v);
		if(Double.isInfinite(p)) {
			return m;
		}
		double u=0;
		for(int i=0; i<v.length; i++) {
			u += Math.exp((v[i]-m)*p);
		}
		u = m + (1/p)*Math.log(u);
		return u;
	}
	
}
