package de.deltatree.tools.google.tf;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Collectors;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import com.google.common.io.Resources;

public class CMDLineMain {

	@Option(name = "--scanDir", required = true, help = true)
	private File scanDir;

	@Option(name = "--temp", required = false, help = true)
	private File tempDir = new File("./modelDir");

	public static void main(String[] args) throws IOException {
		new CMDLineMain().doMain(args);
	}

	public void doMain(String[] args) throws IOException {
		CmdLineParser parser = new CmdLineParser(this);
		try {
			parser.parseArgument(args);

		} catch (CmdLineException e) {
			System.err.println(e.getMessage());
			return;
		}

		checkDirectory(this.scanDir);
		this.tempDir.mkdirs();
		checkDirectory(this.tempDir);

		FileOutputStream to = new FileOutputStream(new File(this.tempDir, "saved_model.pb"));
		Resources.copy(Resources.getResource("saved_model.pb"), to);
		to.close();

		TFModelClassificator classificator = new TFModelClassificator(this.tempDir.toPath());

		for (File f : scanDir.listFiles()) {
			try {
				if (f.isFile() && (f.getName().endsWith(".jpg") || f.getName().endsWith(".png")
						|| f.getName().endsWith(".gif"))) {
					Map<String, Float> classify = classificator.classify(f);

					LinkedHashMap<String, Float> classified = classify.entrySet().stream()
							.sorted(Collections.reverseOrder(Map.Entry.comparingByValue())).collect(Collectors
									.toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e2, LinkedHashMap::new));

					String identifiedClazz = classified.entrySet().stream().findFirst().get().getKey();

					System.out.println("OK   " + f.getName() + " -> " + classify + " -> " + identifiedClazz);
				}
			} catch (Exception e) {
				System.out.println("ERROR" + f.getName() + " -> " + e.getMessage());
			}
		}
	}

	private void checkDirectory(File scanDirIn) {
		if (!scanDirIn.exists()) {
			System.err.println(scanDirIn.getAbsolutePath() + " does not exist!");
			System.exit(1);
		}

		if (!scanDirIn.isDirectory()) {
			System.err.println(scanDirIn.getAbsolutePath() + " is no directory!");
			System.exit(1);
		}
	}
}
