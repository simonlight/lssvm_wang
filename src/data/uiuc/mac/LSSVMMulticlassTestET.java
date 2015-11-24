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

public class LSSVMMulticlassTestET {
	
	
	
	public static void main(String[] args) {
		
		String dataSource= "big";//local or other things
		String gazeType = "ferrari";

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
//		String[] classes = {"bicycle"};
//		int[] scaleCV = {90};
		
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
				String listTrainPath =  sourceDir+"example_files/"+scale+"/"+className+"_train_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";
				List<TrainingSample<BagMIL>> listTrain = BagReader.readBagMIL(listTrainPath, numWords, dataSource);
//				String listTestPath =  sourceDir+"example_files/"+scale+"/"+className+"_val_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";
//				List<TrainingSample<BagMIL>> listTest = BagReader.readBagMIL(listTestPath, numWords, dataSource); 
					
				for(double epsilon : epsilonCV) {
			    	for(double lambda : lambdaCV) {
			    		for(double tradeoff : tradeoffCV){    		    			
	
			    			List<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>> exampleTrain = new ArrayList<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>>();	
			    			for(int i=0; i<listTrain.size(); i++) {
								exampleTrain.add(new STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>(new LatentRepresentation<BagMIL, Integer>(listTrain.get(i).sample,0), listTrain.get(i).label));
			    			}
	
//							List<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>> exampleTest = new ArrayList<STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>>();
//							for(int i=0; i<listTest.size(); i++) {
//								exampleTest.add(new STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer>(new LatentRepresentation<BagMIL, Integer>(listTest.get(i).sample,0), listTest.get(i).label));    			
//							}
							
							LSSVMMulticlassFastBagMILET lsvm = new LSSVMMulticlassFastBagMILET();
							
							File fileClassifier = new File(classifierFolder + "/" + className + "/"+ 
									className + "_" + scale + "_"+epsilon+"_"+lambda + 
									"_"+tradeoff+"_"+cpmax+"_"+cpmin+"_"+epochsLatentMax+"_"+epochsLatentMin +
									"_"+optim+"_"+numWords+".lssvm");
							fileClassifier.getAbsoluteFile().getParentFile().mkdirs();
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
							else {
								System.out.println("training classifier " + fileClassifier.getAbsolutePath());
		    		    		lsvm.setOptim(optim);
		    		    		lsvm.setEpochsLatentMax(epochsLatentMax);
		    		    		lsvm.setEpochsLatentMin(epochsLatentMin);
		    		    		lsvm.setCpmax(cpmax);
		    		    		lsvm.setCpmin(cpmin);
		    		    		lsvm.setLambda(lambda);
		    		    		lsvm.setEpsilon(epsilon);
								lsvm.setGazeType(gazeType);
									
								lsvm.setLossDict(sourceDir+"ETLoss_dict/"+"ETLOSS+_"+scale+".loss");
								lsvm.setTradeOff(tradeoff);
								lsvm.setHnorm(hnorm);
								lsvm.setCurrentClass(className);
								//Initialize the region by fixations
								for(STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer> ts : exampleTrain){
									ts.input.h = lsvm.getGazeInitRegion(ts, scale, initializedType);
								}
							
								lsvm.train(exampleTrain);
							}
							
														
		    				if (saveClassifier){
			    				// save classifier
								
			    				ObjectOutputStream oos = null;
								try {
									oos = new ObjectOutputStream(new FileOutputStream(fileClassifier.getAbsolutePath()));
									oos.writeObject(lsvm);
								} 
								catch (FileNotFoundException e) {
									e.printStackTrace();
								} 
								catch (IOException e) {
									e.printStackTrace();
								}
								finally {
									try {
										if(oos != null) {
											oos.flush();
											oos.close();
										}
									} catch (IOException e) {
										e.printStackTrace();
									}
								}
								System.out.println("wrote classifier successfully!");
							}
							
							//metric file
							File trainMetricFile=new File(metricFolder+"/metric_train_"+tradeoff+"_"+scale+"_"+epsilon+"_"+lambda+"_"+className+".txt");
							trainMetricFile.getAbsoluteFile().getParentFile().mkdirs();
		    				double ap_train = lsvm.testAPRegion(exampleTrain, trainMetricFile);
		    				System.out.println("ap train:"+ap_train);
		    				//		    				
//		    				File testMetricFile=new File(metricFolder+"/metric_test_"+tradeoff+"_"+scale+"_"+epsilon+"_"+lambda+"_"+className+".txt");
//		    				testMetricFile.getAbsoluteFile().getParentFile().mkdirs();
//		    				double ap_test = lsvm.testAPRegion(exampleTest, testMetricFile);
//							
//		    				//write ap 
//		    				try {
//								BufferedWriter out = new BufferedWriter(new FileWriter(resultFilePath, true));
//								out.write(className+" "+String.valueOf(tradeoff)+" "+scale+" "+lambda+" "+epsilon+" "+ap_test+" "+ap_train+"\n");
//								out.flush();
//								out.close();
//								
//							} catch (IOException e) {
//								// TODO Auto-generated catch block
//								e.printStackTrace();
//							}
//	
//		    				System.err.println(className + " test "+ String.valueOf(tradeoff)+" "+className + " scale= " + scale + " ap= " + ap_test + " lambda= " + lambda + " epsilon= " + epsilon);
//							System.out.println("\n");
//							
//		    				System.err.format("train:%s category:%s scale:%s lambda:%s epsilon:%s %n ", ap_train, className, scale, lambda, epsilon); 
							

						}
			    	}
			    }
			}
		}
	}
	
	
}
