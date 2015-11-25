package data.uiuc.mac;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import latent.LatentRepresentation;
import latent.lssvm.multiclass.LSSVMMulticlassFastBagMILET;
import latent.variable.BagMIL;
import struct.STrainingSample;
import data.io.BagReader;
import fr.lip6.jkernelmachines.type.TrainingSample;

public class LSSVMMulticlassEvaluationET {
	
	public static void main(String[] args) {
		
		String dataSource= "big";//local or other things
		String gazeType = "ferrari;

		String sourceDir = new String();
		String resDir = new String();

		if (dataSource=="local" && gazeType == "ferrari"){
			sourceDir = "/local/wangxin/Data/ferrari_gaze/";
			resDir = "/local/wangxin/results/ferrari_gaze/std_et/";
			gazeType = "ferrari";
		}
		else if (dataSource=="big" && gazeType == "ferrari"){
			sourceDir = "/home/wangxin/Data/ferrari_gaze/";
			resDir = "/home/wangxin/results/ferrari_gaze/std_et/";
			gazeType = "ferrari";
		}
		else if (dataSource=="local" && gazeType == "stefan"){
			sourceDir = "/local/wangxin/Data/gaze_voc_actions_stefan/";
			resDir = "/local/wangxin/results/stefan_gaze/std_et/";
			gazeType = "stefan";
			
		}
		else if (dataSource=="big" && gazeType == "stefan"){
			sourceDir = "/home/wangxin/Data/gaze_voc_actions_stefan/";
			resDir = "/home/wangxin/results/stefan_gaze/std_et/";
			gazeType = "stefan";
			
		}
	
		String initializedType = ".";//+0,+-,or other things
		boolean hnorm = false;
		
		String taskName = "java_std_et/";
		
		String resultFolder = resDir+taskName;
		
		String resultFilePath = resultFolder + "ap_summary.txt";
		String metricFolder = resultFolder + "metric/";
		String classifierFolder = resultFolder + "classifier/";
		String scoreFolder = resultFolder + "score/";
	
		String[] classes = {args[0]};
		int[] scaleCV = {Integer.valueOf(args[1])};
//		String[] classes = {"boat"};
//		int[] scaleCV = {50};
		
	    double[] lambdaCV = {1e-4};
	    double[] epsilonCV = {1e-3};
	
	    double[] tradeoffCV = {0, 0.1,0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1.0};
			    
	//	    String[] classes = {"dog", "cat", "motorbike", "boat" ,"aeroplane" ,"horse" ,"cow" ,"sofa", "diningtable" ,"bicycle"};
	    
		int optim = 1;
		int epochsLatentMax = 5000;
		int epochsLatentMin = 2;
		int cpmax = 5000;
		int cpmin = 2;
		int numWords = 2048;
		boolean saveClassifier = true;
	    boolean loadClassifier = true;
	    
		System.out.println("experiment detail: "
				+ "\nsourceDir:\t "+sourceDir
				+ "\nresDir:\t"+resDir
				+ "\ngaze type:\t"+gazeType
				+ "\ninitilaize type:\t"+initializedType
				+ "\nhnorm:\t"+Boolean.toString(hnorm)
				+ "\ntask name:\t"+taskName
				+ "\nclasses CV:\t"+Arrays.toString(classes)
				+ "\nscale CV:\t"+Arrays.toString(scaleCV)
				+ "\nlambda CV:\t" + Arrays.toString(lambdaCV)
				+ "\nepsilon CV:\t" + Arrays.toString(epsilonCV)
				+ "\ntradeoff CV:\t"+Arrays.toString(tradeoffCV)
				+ "\noptim:\t"+optim
				+ "\nepochsLatentMax:\t"+epochsLatentMax
				+ "\nepochsLatentMin:\t"+epochsLatentMin
				+ "\ncpmax:\t"+cpmax
				+ "\ncpmin:\t"+cpmin
				+ "\nnumWords:\t"+numWords
				+ "\nsaveClassifier:\t"+Boolean.toString(saveClassifier)
			    + "\nloadClassifier:\t"+Boolean.toString(loadClassifier)
			    );
	    
		
	    for(String className: classes){
	    	for(int scale : scaleCV) {
	    		String listValPath =  sourceDir+"example_files/"+scale+"/"+className+"_valval_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";
				List<TrainingSample<BagMIL>> listVal = BagReader.readBagMIL(listValPath, numWords, dataSource); 
	    		String listTestPath =  sourceDir+"example_files/"+scale+"/"+className+"_valtest_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";
				List<TrainingSample<BagMIL>> listTest = BagReader.readBagMIL(listTestPath, numWords, dataSource); 
					
				for(double epsilon : epsilonCV) {
			    	for(double lambda : lambdaCV) {
			    		for(double tradeoff : tradeoffCV){    		    			
	
			    			List<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>> exampleVal = new ArrayList<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>>();	
			    			for(int i=0; i<listVal.size(); i++) {
			    				exampleVal.add(new STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>(new LatentRepresentation<BagMIL, Integer>(listVal.get(i).sample,0), listVal.get(i).label));
			    			}
//	
							List<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>> exampleTest = new ArrayList<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>>();
							for(int i=0; i<listTest.size(); i++) {
								exampleTest.add(new STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>(new LatentRepresentation<BagMIL, Integer>(listTest.get(i).sample,0), listTest.get(i).label));    			
							}
							
							LSSVMMulticlassFastBagMILET lsvm = new LSSVMMulticlassFastBagMILET();
							
							File fileClassifier = new File(classifierFolder + "/" + className + "/"+ 
									className + "_" + scale + "_"+epsilon+"_"+lambda + 
									"_"+tradeoff+"_"+cpmax+"_"+cpmin+"_"+epochsLatentMax+"_"+epochsLatentMin +
									"_"+optim+"_"+numWords+".lssvm");
							if (loadClassifier && fileClassifier.exists()){
								ObjectInputStream ois;
								System.out.println("read classifier " + fileClassifier.getAbsolutePath());
								try {
									ois = new ObjectInputStream(new FileInputStream(fileClassifier.getAbsolutePath()));
									lsvm = (LSSVMMulticlassFastBagMILET) ois.readObject();
									lsvm.showParameters();
								} catch (FileNotFoundException e) {
									// TODO Auto-generated catch block
									e.printStackTrace();
								} catch (IOException e) {
									// TODO Auto-generated catch block
									e.printStackTrace();
								} catch (ClassNotFoundException e) {
									// TODO Auto-generated catch block
									e.printStackTrace();
								}
							}
							
//							//val metric file
							File valMetricFile=new File(metricFolder+className+"/metric_valval_"+tradeoff+"_"+scale+"_"+epsilon+"_"+lambda+"_"+className+".txt");
							valMetricFile.getAbsoluteFile().getParentFile().mkdirs();
		    				double ap_val = lsvm.testAPRegion(exampleVal, valMetricFile);
		    				System.out.println("ap train:"+ap_val);
		    				//test metric file		    				
		    				File testMetricFile=new File(metricFolder+className+"/metric_valtest_"+tradeoff+"_"+scale+"_"+epsilon+"_"+lambda+"_"+className+".txt");
		    				testMetricFile.getAbsoluteFile().getParentFile().mkdirs();
		    				double ap_test = lsvm.testAPRegion(exampleTest, testMetricFile);
		    				System.out.println("ap test:"+ap_test);
		    				
		    				//write ap 
		    				try {
								BufferedWriter out = new BufferedWriter(new FileWriter(resultFilePath, true));
								out.write(className+" "+String.valueOf(tradeoff)+" "+scale+" "+lambda+" "+epsilon+" "+ap_val+" "+ap_test+"\n");
								out.flush();
								out.close();
								
							} catch (IOException e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							}
	
		    				System.err.format("val:%s category:%s scale:%s lambda:%s epsilon:%s %n ", ap_val, className, scale, lambda, epsilon); 
							System.out.println("\n");
		    				System.err.format("test:%s category:%s scale:%s lambda:%s epsilon:%s %n ", ap_test, className, scale, lambda, epsilon); 
							

						}
			    	}
			    }
			}
		}
	}
	
	
}
