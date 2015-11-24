package Test;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import solver.*;
public class str_eq {
	public static void main(String[] args) throws FileNotFoundException, IOException, ClassNotFoundException {
		float i =10.0f;
		System.out.format("the calue is %s%n",i);
	
		
//		ObjectInputStream is = new ObjectInputStream(new FileInputStream("/local/wangxin/Data/ferrari_gaze/ETLoss_dict/ETLOSS+_50.loss"));  
//        HashMap<String[] , Double>  temp = (HashMap<String[], Double> ) is.readObject();// 从流中读取User的数据  
//        System.out.println(temp.get("horse_2010_001856_3_4.txt"));
//        is.close();
		
}
	public void mosekQuad(){
//		double[][] gram, List<Double> lc, double c
		double[][] gram= new double[2][2];
		List<Double> lc = new ArrayList<Double>();
		gram[0][0] = 1.0;
		gram[0][1] = 0.0;
		gram[1][0] = 0.0;
		gram[1][1] = 1.0;
		lc.add(2.0);
		lc.add(2.0);
		
		double[] alpha = solver.MosekSolver.solveQP(gram, lc, 1.0);
		for(double a: alpha){
			System.out.println(a);
		}
	}
}