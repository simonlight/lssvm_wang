package data.io;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import latent.variable.BagMIL;
import util.VectorOp;
import extern.pca.PrincipalComponentAnalysis;
import fr.lip6.jkernelmachines.type.TrainingSample;

public class BagReader {
	
	public static List<TrainingSample<BagMIL>> readBagMIL(String file, int dim) {
		List<TrainingSample<BagMIL>> list = readBagMIL(new File(file), dim, true, null);
		return list;
	}
	
	public static List<TrainingSample<BagMIL>> readBagMIL(String file, int dim, boolean bias, PrincipalComponentAnalysis pca) {
		List<TrainingSample<BagMIL>> list = readBagMIL(new File(file), dim, bias, pca);
		return list;
	}
	
	public static List<TrainingSample<BagMIL>> readBagMIL(File file, int dim, boolean bias, PrincipalComponentAnalysis pca) {
		List<TrainingSample<BagMIL>> list = null;
		if(file.exists()) {
			try {
				
				System.out.println("read bag: " + file.getAbsolutePath() + "\tdim= " + dim);
				InputStream ips = new FileInputStream(file); 
				InputStreamReader ipsr = new InputStreamReader(ips);
				BufferedReader br = new BufferedReader(ipsr);
				
				list = new ArrayList<TrainingSample<BagMIL>>();
				int nbInstancesAll = 0;
				String ligne;
				ligne=br.readLine();
				System.out.println(ligne);
				int nbBag = Integer.parseInt(ligne);
				//test!!!!!!!!!!!!!
				nbBag=2;
				//
				for(int i=0; i<nbBag; i++) {
					
					System.out.print(".");
					if(i>0 && i % 100 == 0) System.out.print(i);
					ligne=br.readLine();
					//System.out.println(ligne);
					StringTokenizer st = new StringTokenizer(ligne);
					String name = st.nextToken();
					int label = Integer.parseInt(st.nextToken());
					int nbInstances = Integer.parseInt(st.nextToken());
					//System.out.println("name: " + name + "\tlabel: " + label + "\tnbInstances: " + nbInstances);
					BagMIL bag = new BagMIL();
					bag.setName(name);
					
					for(int j=0; j<nbInstances; j++) {
						
						String filefeature = st.nextToken();
						
						bag.addFileFeature(filefeature);
						
						double[] feature = readFeature(new File(filefeature));
						
						if(feature.length != dim) {
							System.out.println("ERROR features - dim= " + feature.length + " != " + dim);
							System.out.println("file " + filefeature);
							feature = null;
							System.exit(0);
						}
						bag.addFeature(feature);
					}
					
					nbInstancesAll += bag.getFeatures().size();
					list.add(new TrainingSample<BagMIL>(bag,label));
				}
				
				br.close();
				System.out.println("\nnb bags: " + list.size() + "\tnb instances: " + nbInstancesAll + "\tnb moyen instances: " + (nbInstancesAll/list.size()));
			
				for(TrainingSample<BagMIL> ts : list) {
					for(int i=0; i<ts.sample.getFeatures().size(); i++) {
						VectorOp.normL2(ts.sample.getFeature(i));
						if(pca != null) ts.sample.setFeature(i, pca.sampleToEigenSpace(ts.sample.getFeature(i)));
						if(bias) ts.sample.setFeature(i,VectorOp.addValeur(ts.sample.getFeature(i),1));
					}
				}
				
			}
			catch (IOException e) {
				System.out.println("Error parsing file " + file);
				return null;
			}
		}
		else {
			System.out.println("file " + file.getAbsolutePath() + " does not exist");
		}
		return list;
	}
	
	public static List<TrainingSample<BagMIL>> readBagMILNoFeatures(File file) {
		List<TrainingSample<BagMIL>> list = null;
		if(file.exists()) {
			try {
				System.out.println("read bag: " + file.getAbsolutePath());
				InputStream ips = new FileInputStream(file); 
				InputStreamReader ipsr = new InputStreamReader(ips);
				BufferedReader br = new BufferedReader(ipsr);
				
				list = new ArrayList<TrainingSample<BagMIL>>();
				
				String ligne;
				ligne=br.readLine();
				int nbBag = Integer.parseInt(ligne);
				for(int i=0; i<nbBag; i++) {
					System.out.print(".");
					if(i>0 && i % 100 == 0) System.out.print(i);
					ligne=br.readLine();
					StringTokenizer st = new StringTokenizer(ligne);
					String name = st.nextToken();
					int label = Integer.parseInt(st.nextToken());
					BagMIL bag = new BagMIL();
					bag.setName(name);
					list.add(new TrainingSample<BagMIL>(bag,label));
				}
				br.close();
				System.out.println("\nnb bags: " + list.size());
			
				for(TrainingSample<BagMIL> ts : list) {
					for(int i=0; i<ts.sample.getFeatures().size(); i++) {
						VectorOp.normL2(ts.sample.getFeature(i));
						ts.sample.setFeature(i,VectorOp.addValeur(ts.sample.getFeature(i),1));
					}
				}
				
			}
			catch (IOException e) {
				System.out.println("Error parsing file " + file);
				return null;
			}
		}
		else {
			System.out.println("file " + file.getAbsolutePath() + " does not exist");
		}
		return list;
	}
	
	private static double[] readFeature(File file) {
		
		double[] feature = null;
		if(file.exists()) {
			List<Double> l = new ArrayList<Double>();
			
			try {
				InputStream ips = new FileInputStream(file); 
				InputStreamReader ipsr = new InputStreamReader(ips);
				BufferedReader br = new BufferedReader(ipsr);
				
				String ligne;
				while ((ligne=br.readLine()) != null){
					l.add(Double.parseDouble(ligne));
				}
				
				br.close();
			}
			catch (IOException e) {
				System.out.println("Error parsing file " + file);
			}
			
			feature = new double[l.size()];
			for(int i=0; i<l.size(); i++) {
				feature[i] = l.get(i);
			}
			
			//System.out.println("PPMI - read feature: " + file.getAbsoluteFile() + "\tdim: " + feature.length);
		}
		else {
			System.out.println("Features file " + file.getAbsolutePath() + " does not exist");
			System.exit(0);
		}
		
		
		return feature;
		
	}
}
