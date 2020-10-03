package de.deltatree.tools.google.tf;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import org.tensorflow.EagerSession;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.impl.buffer.nio.NioDataBufferFactory;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Reshape;
import org.tensorflow.op.image.EncodeJpeg;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TString;
import org.tensorflow.types.TUint8;

import com.google.common.collect.Maps;

public class TFModelClassificator {

	private final SavedModelBundle modelBundle;

	public TFModelClassificator(Path path) {
		this.modelBundle = SavedModelBundle.load(path.toFile().getAbsolutePath(), "serve");
	}

	public Map<String, Float> classify(File file) throws Exception {
		Map<String, Float> result = Maps.newHashMap();

		Tensor<TUint8> makeImageTensor = makeImageTensor(file);

		try (EagerSession eagerSession = EagerSession.create()) {
			Ops tf = Ops.create(eagerSession);

			Ops tf_deltatree = tf;
			Constant<TUint8> constant = tf_deltatree.constant(makeImageTensor);
			EncodeJpeg encodeJpeg = tf.image.encodeJpeg(constant);

			Tensor<TString> asTensor = encodeJpeg.asTensor();

			Constant<TString> tensor = Constant.create(tf.scope(), asTensor);

			Tensor<TInt32> tensorOf = TInt32.tensorOf(tensor.data().shape());
			tensorOf.data().setInt(1);
			Constant<TInt32> shape = Constant.create(tf.scope(), tensorOf);

			Reshape<TString> reshape = tf.reshape(tensor, shape);

			Session session = modelBundle.session();

			List<Tensor<?>> run = session.runner().feed("Placeholder", reshape.asTensor())
					.feed("Placeholder_1", TString.vectorOf(file.getName())).fetch("Tile", 0).fetch("scores").run();

			@SuppressWarnings("unchecked")
			Tensor<TFloat32> scores = (Tensor<TFloat32>) run.get(1);
			@SuppressWarnings("unchecked")
			Tensor<TString> tile = (Tensor<TString>) run.get(0);
			NdArray<String> ndArray = tile.data().get(0);

			for (int i = 0; i < ndArray.size(); i++) {
				result.put(ndArray.getObject(i), scores.data().get(0).getFloat(i));
			}
		}
		return result;
	}

	private static Tensor<TUint8> makeImageTensor(File file) throws IOException {
		BufferedImage img = ImageIO.read(file);
		if (img.getType() != BufferedImage.TYPE_3BYTE_BGR) {
			BufferedImage newImage = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
			Graphics2D g = newImage.createGraphics();
			g.drawImage(img, 0, 0, img.getWidth(), img.getHeight(), null);
			g.dispose();
			img = newImage;
		}

		byte[] data = ((DataBufferByte) img.getData().getDataBuffer()).getData();
		bgr2rgb(data);
		final long CHANNELS = 3;
		long[] shape = new long[] { img.getHeight(), img.getWidth(), CHANNELS };

		return TUint8.tensorOf(Shape.of(shape), NioDataBufferFactory.create(ByteBuffer.wrap(data)));
	}

	private static void bgr2rgb(byte[] data) {
		for (int i = 0; i < data.length; i += 3) {
			byte tmp = data[i];
			data[i] = data[i + 2];
			data[i + 2] = tmp;
		}
	}
}
