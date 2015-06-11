package graph.variable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DebugNormProduct {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		int[][] adjMatrix = {{0,1,1,0}, {1,0,0,1}, {1,0,0,1},{0,1,1,0}};
		System.out.println("adj matrix");
		for(int[] t : adjMatrix) {
			System.out.println(Arrays.toString(t));
		}
		
		int[][] cPair	= adjMatrix.clone();
		double[][] cCond	= new double[adjMatrix.length][adjMatrix.length];
		double[] cLocal 	= new double[adjMatrix.length];
		
		double[][] logLocalPotentials = new double[adjMatrix.length][];
		for(int i=0; i<adjMatrix.length; i++) {
			logLocalPotentials[i] = new double[i+2];
		}
		
		List<double[]> logPairPotentials = new ArrayList<double[]>();
	
		NormProduct.inference(adjMatrix, logLocalPotentials, cPair, cCond, cLocal);
		
		System.out.println("END");
	}

}
