/*
   Copyright 2012 Harri Smatt

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

package fi.harism.wallpaper.flipcube;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import android.content.Context;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.widget.Toast;

/**
 * Renderer class.
 */
public final class FlipCubeRenderer implements GLSurfaceView.Renderer {

	// Screen aspect ratio.
	private final float[] mAspectRatio = new float[2];
	private ByteBuffer mBufferCubeFilled;
	private ByteBuffer mBufferCubeLines;
	private ByteBuffer mBufferQuad;
	private Context mContext;
	private final FlipCubeFbo mFbo = new FlipCubeFbo();
	private boolean mFlipTexture = true;
	private long mFlipTime;
	private float[] mMatrixProj = new float[16];
	private long mRenderTime;
	private final boolean[] mShaderCompilerSupport = new boolean[1];
	private final FlipCubeShader mShaderFill = new FlipCubeShader();
	private final FlipCubeShader mShaderFlip = new FlipCubeShader();
	private final FlipCubeShader mShaderLine = new FlipCubeShader();
	private final FlipCubeShader mShaderPage = new FlipCubeShader();
	private int mWidth, mHeight;

	/**
	 * Default constructor.
	 */
	public FlipCubeRenderer(Context context) {
		mContext = context;

		// Full view quad buffer.
		final byte[] QUAD = { -1, 1, -1, -1, 1, 1, 1, -1 };
		mBufferQuad = ByteBuffer.allocateDirect(8);
		mBufferQuad.put(QUAD).position(0);

		// Vertex and normal data plus indices arrays.
		final byte[][] CUBEVERTICES = { { -1, 1, 1 }, { -1, -1, 1 },
				{ 1, 1, 1 }, { 1, -1, 1 }, { -1, 1, -1 }, { -1, -1, -1 },
				{ 1, 1, -1 }, { 1, -1, -1 } };
		final byte[][] CUBENORMALS = { { 0, 0, 1 }, { 0, 0, -1 }, { -1, 0, 0 },
				{ 1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 } };
		final int[][][] CUBELINES = { { { 0, 1, 1, 3, 3, 2, 2, 0 }, { 0 } },
				{ { 4, 5, 5, 7, 7, 6, 6, 4 }, { 1 } },
				{ { 0, 4, 4, 5, 5, 1, 1, 0 }, { 2 } },
				{ { 2, 3, 3, 7, 7, 6, 6, 2 }, { 3 } },
				{ { 0, 2, 2, 6, 6, 4, 4, 0 }, { 4 } },
				{ { 1, 3, 3, 7, 7, 5, 5, 1 }, { 5 } } };
		final int[][][] CUBEFILLED = { { { 0, 1, 2, 1, 3, 2 }, { 0 } },
				{ { 6, 7, 4, 7, 5, 4 }, { 1 } },
				{ { 0, 4, 1, 4, 5, 1 }, { 2 } },
				{ { 3, 7, 2, 7, 6, 2 }, { 3 } },
				{ { 4, 0, 6, 0, 2, 6 }, { 4 } },
				{ { 1, 5, 3, 5, 7, 3 }, { 5 } } };

		// Generate lines buffer.
		mBufferCubeLines = ByteBuffer.allocateDirect(6 * 8 * 6);
		for (int i = 0; i < CUBELINES.length; ++i) {
			for (int j = 0; j < CUBELINES[i][0].length; ++j) {
				mBufferCubeLines.put(CUBEVERTICES[CUBELINES[i][0][j]]);
				mBufferCubeLines.put(CUBENORMALS[CUBELINES[i][1][0]]);
			}
		}
		mBufferCubeLines.position(0);

		// Generate fill buffer.
		mBufferCubeFilled = ByteBuffer.allocateDirect(6 * 6 * 6);
		for (int i = 0; i < CUBEFILLED.length; ++i) {
			for (int j = 0; j < CUBEFILLED[i][0].length; ++j) {
				mBufferCubeFilled.put(CUBEVERTICES[CUBEFILLED[i][0][j]]);
				mBufferCubeFilled.put(CUBENORMALS[CUBEFILLED[i][1][0]]);
			}
		}
		mBufferCubeFilled.position(0);
	}

	/**
	 * Fast inverse-transpose matrix calculation. See
	 * http://content.gpwiki.org/index.php/MathGem:Fast_Matrix_Inversion for
	 * more information. Only difference is that we do transpose at the same
	 * time and therefore we don't transpose upper-left 3x3 matrix leaving it
	 * intact. Also T is written into lowest row of destination matrix instead
	 * of last column.
	 * 
	 * @param dst
	 *            Destination matrix
	 * @param dstOffset
	 *            Destination matrix offset
	 * @param src
	 *            Source matrix
	 * @param srcOffset
	 *            Source matrix offset
	 */
	private void invTransposeM(float[] dst, int dstOffset, float[] src,
			int srcOffset) {
		android.opengl.Matrix.setIdentityM(dst, dstOffset);

		// Copy top-left 3x3 matrix into dst matrix.
		dst[dstOffset + 0] = src[srcOffset + 0];
		dst[dstOffset + 1] = src[srcOffset + 1];
		dst[dstOffset + 2] = src[srcOffset + 2];
		dst[dstOffset + 4] = src[srcOffset + 4];
		dst[dstOffset + 5] = src[srcOffset + 5];
		dst[dstOffset + 6] = src[srcOffset + 6];
		dst[dstOffset + 8] = src[srcOffset + 8];
		dst[dstOffset + 9] = src[srcOffset + 9];
		dst[dstOffset + 10] = src[srcOffset + 10];

		// Calculate -(Ri dot T) into last row.
		dst[dstOffset + 3] = -(src[srcOffset + 0] * src[srcOffset + 12]
				+ src[srcOffset + 1] * src[srcOffset + 13] + src[srcOffset + 2]
				* src[srcOffset + 14]);
		dst[dstOffset + 7] = -(src[srcOffset + 4] * src[srcOffset + 12]
				+ src[srcOffset + 5] * src[srcOffset + 13] + src[srcOffset + 6]
				* src[srcOffset + 14]);
		dst[dstOffset + 11] = -(src[srcOffset + 8] * src[srcOffset + 12]
				+ src[srcOffset + 9] * src[srcOffset + 13] + src[srcOffset + 10]
				* src[srcOffset + 14]);
	}

	/**
	 * Loads String from raw resources with given id.
	 */
	private String loadRawString(int rawId) throws Exception {
		InputStream is = mContext.getResources().openRawResource(rawId);
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		byte[] buf = new byte[1024];
		int len;
		while ((len = is.read(buf)) != -1) {
			baos.write(buf, 0, len);
		}
		return baos.toString();
	}

	@Override
	public void onDrawFrame(GL10 unused) {

		// If shader compiler not supported return immediately.
		if (!mShaderCompilerSupport[0]) {
			// Clear view buffer.
			GLES20.glClearColor(0f, 0f, 0f, 1f);
			GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
			return;
		}

		// Disable unnecessary OpenGL flags.
		GLES20.glDisable(GLES20.GL_DEPTH_TEST);
		GLES20.glDisable(GLES20.GL_CULL_FACE);

		long time = SystemClock.uptimeMillis();
		if (time - mRenderTime > mFlipTime) {
			mFbo.bind();
			if (mFlipTexture) {
				mFbo.bindTexture(1);
			} else {
				mFbo.bindTexture(0);
			}
			renderPage();
			mFlipTexture = !mFlipTexture;

			mRenderTime = time;
			mFlipTime = 200 + (long) (300 * Math.sin(time % 30000 / 30000f
					* Math.PI));
		}
		float t = (time - mRenderTime) / (float) mFlipTime;

		// Copy offscreen buffer to screen.
		GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
		GLES20.glViewport(0, 0, mWidth, mHeight);

		mShaderFlip.useProgram();
		int sTop = mShaderFlip.getHandle("sTop");
		int sBottom = mShaderFlip.getHandle("sBottom");
		int uFlipPos = mShaderFlip.getHandle("uFlipPos");
		int aPosition = mShaderFlip.getHandle("aPosition");

		GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
		GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, mFbo.getTexture(0));
		GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
		GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, mFbo.getTexture(1));

		if (mFlipTexture) {
			GLES20.glUniform1i(sTop, 1);
			GLES20.glUniform1i(sBottom, 0);
		} else {
			GLES20.glUniform1i(sTop, 0);
			GLES20.glUniform1i(sBottom, 1);
		}

		GLES20.glVertexAttribPointer(aPosition, 2, GLES20.GL_BYTE, false, 0,
				mBufferQuad);
		GLES20.glEnableVertexAttribArray(aPosition);

		GLES20.glUniform1f(uFlipPos, 1f - t);

		GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

	}

	@Override
	public void onSurfaceChanged(GL10 unused, int width, int height) {
		mWidth = width;
		mHeight = height;

		// Store view aspect ratio.
		mAspectRatio[0] = (float) Math.max(mWidth, mHeight) / mHeight;
		mAspectRatio[1] = (float) Math.max(mWidth, mHeight) / mWidth;

		Matrix.orthoM(mMatrixProj, 0, -mAspectRatio[0], mAspectRatio[0],
				-mAspectRatio[1], mAspectRatio[1], .1f, 20f);

		mFbo.init(mWidth, mHeight, 2);
		mFbo.bindTexture(0);
		renderPage();
	}

	@Override
	public void onSurfaceCreated(GL10 unused, EGLConfig config) {
		// Check if shader compiler is supported.
		GLES20.glGetBooleanv(GLES20.GL_SHADER_COMPILER, mShaderCompilerSupport,
				0);

		// If not, show user an error message and return immediately.
		if (mShaderCompilerSupport[0] == false) {
			String msg = mContext.getString(R.string.error_shader_compiler);
			showError(msg);
			return;
		}

		// Load vertex and fragment shaders.
		try {
			String vertexSource, fragmentSource;
			vertexSource = loadRawString(R.raw.page_vs);
			fragmentSource = loadRawString(R.raw.page_fs);
			mShaderPage.setProgram(vertexSource, fragmentSource);
			vertexSource = loadRawString(R.raw.flip_vs);
			fragmentSource = loadRawString(R.raw.flip_fs);
			mShaderFlip.setProgram(vertexSource, fragmentSource);
			vertexSource = loadRawString(R.raw.line_vs);
			fragmentSource = loadRawString(R.raw.line_fs);
			mShaderLine.setProgram(vertexSource, fragmentSource);
			vertexSource = loadRawString(R.raw.fill_vs);
			fragmentSource = loadRawString(R.raw.fill_fs);
			mShaderFill.setProgram(vertexSource, fragmentSource);
		} catch (Exception ex) {
			showError(ex.getMessage());
		}
	}

	/**
	 * Renders page contents into currently active FBO.
	 */
	private void renderPage() {
		// Render page background.
		mShaderPage.useProgram();
		int uAspectRatio = mShaderPage.getHandle("uAspectRatio");
		int aPosition = mShaderPage.getHandle("aPosition");

		GLES20.glUniform2fv(uAspectRatio, 1, mAspectRatio, 0);
		GLES20.glVertexAttribPointer(aPosition, 2, GLES20.GL_BYTE, false, 0,
				mBufferQuad);
		GLES20.glEnableVertexAttribArray(aPosition);

		GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

		// Local rotation values.
		long time = SystemClock.uptimeMillis();
		float rx = ((time % 10000) / 10000f) * 360f;
		float ry = ((time % 12000) / 12000f) * 360f;
		float rz = ((time % 15000) / 15000f) * 360f;

		final float[] modelViewProjM = new float[16];
		Matrix.setIdentityM(modelViewProjM, 0);
		final float[] scaleM = new float[16];
		Matrix.setIdentityM(scaleM, 0);
		Matrix.scaleM(scaleM, 0, 0.5f, 0.5f, 0.5f);
		final float[] rotateM = new float[16];
		setRotateM(rotateM, 0, rx, ry, rz);
		final float[] translateM = new float[16];
		Matrix.setIdentityM(translateM, 0);
		Matrix.translateM(translateM, 0, 0f, 0f, -1f);

		Matrix.multiplyMM(modelViewProjM, 0, scaleM, 0, modelViewProjM, 0);
		Matrix.multiplyMM(modelViewProjM, 0, rotateM, 0, modelViewProjM, 0);
		Matrix.multiplyMM(modelViewProjM, 0, translateM, 0, modelViewProjM, 0);

		// View rotation values.
		rx = ((time % 15000) / 15000f) * 360f;
		ry = ((time % 18000) / 18000f) * 360f;
		rz = ((time % 20000) / 20000f) * 360f;
		setRotateM(rotateM, 0, rx, ry, rz);
		Matrix.translateM(translateM, 0, 0.2f, 0f, -5f);

		Matrix.multiplyMM(modelViewProjM, 0, rotateM, 0, modelViewProjM, 0);
		Matrix.multiplyMM(modelViewProjM, 0, translateM, 0, modelViewProjM, 0);

		final float[] normalM = new float[16];
		invTransposeM(normalM, 0, modelViewProjM, 0);

		Matrix.multiplyMM(modelViewProjM, 0, mMatrixProj, 0, modelViewProjM, 0);

		// Render cube lines.
		mShaderLine.useProgram();
		int uModelViewProjM = mShaderLine.getHandle("uModelViewProjM");
		int uNormalM = mShaderLine.getHandle("uNormalM");
		aPosition = mShaderLine.getHandle("aPosition");
		int aNormal = mShaderLine.getHandle("aNormal");

		GLES20.glUniformMatrix4fv(uModelViewProjM, 1, false, modelViewProjM, 0);
		GLES20.glUniformMatrix4fv(uNormalM, 1, false, normalM, 0);

		mBufferCubeLines.position(0);
		GLES20.glVertexAttribPointer(aPosition, 3, GLES20.GL_BYTE, false, 6,
				mBufferCubeLines);
		GLES20.glEnableVertexAttribArray(aPosition);

		mBufferCubeLines.position(3);
		GLES20.glVertexAttribPointer(aNormal, 3, GLES20.GL_BYTE, false, 6,
				mBufferCubeLines);
		GLES20.glEnableVertexAttribArray(aNormal);

		GLES20.glEnable(GLES20.GL_BLEND);
		GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);

		GLES20.glLineWidth(5f);
		GLES20.glDrawArrays(GLES20.GL_LINES, 0, 8 * 6);

		// Render filled cube.
		mShaderFill.useProgram();
		uModelViewProjM = mShaderFill.getHandle("uModelViewProjM");
		uNormalM = mShaderFill.getHandle("uNormalM");
		aPosition = mShaderFill.getHandle("aPosition");
		aNormal = mShaderFill.getHandle("aNormal");

		GLES20.glUniformMatrix4fv(uModelViewProjM, 1, false, modelViewProjM, 0);
		GLES20.glUniformMatrix4fv(uNormalM, 1, false, normalM, 0);

		mBufferCubeFilled.position(0);
		GLES20.glVertexAttribPointer(aPosition, 3, GLES20.GL_BYTE, false, 6,
				mBufferCubeFilled);
		GLES20.glEnableVertexAttribArray(aPosition);

		mBufferCubeFilled.position(3);
		GLES20.glVertexAttribPointer(aNormal, 3, GLES20.GL_BYTE, false, 6,
				mBufferCubeFilled);
		GLES20.glEnableVertexAttribArray(aNormal);

		GLES20.glEnable(GLES20.GL_CULL_FACE);
		GLES20.glDrawArrays(GLES20.GL_TRIANGLES, 0, 6 * 6);

		GLES20.glDisable(GLES20.GL_CULL_FACE);
		GLES20.glDisable(GLES20.GL_BLEND);
	}

	/**
	 * Calculates rotation matrix into given matrix array.
	 * 
	 * @param m
	 *            Matrix float array
	 * @param offset
	 *            Matrix start offset
	 * @param rx
	 *            Rotation around x axis
	 * @param ry
	 *            Rotation around y axis
	 * @param rz
	 *            Rotation around z axis
	 */
	private void setRotateM(float[] m, int offset, float rx, float ry, float rz) {
		double toRadians = Math.PI * 2 / 360;
		rx *= toRadians;
		ry *= toRadians;
		rz *= toRadians;
		double sin0 = Math.sin(rx);
		double cos0 = Math.cos(rx);
		double sin1 = Math.sin(ry);
		double cos1 = Math.cos(ry);
		double sin2 = Math.sin(rz);
		double cos2 = Math.cos(rz);

		android.opengl.Matrix.setIdentityM(m, offset);

		double sin1_cos2 = sin1 * cos2;
		double sin1_sin2 = sin1 * sin2;

		m[0 + offset] = (float) (cos1 * cos2);
		m[1 + offset] = (float) (cos1 * sin2);
		m[2 + offset] = (float) (-sin1);

		m[4 + offset] = (float) ((-cos0 * sin2) + (sin0 * sin1_cos2));
		m[5 + offset] = (float) ((cos0 * cos2) + (sin0 * sin1_sin2));
		m[6 + offset] = (float) (sin0 * cos1);

		m[8 + offset] = (float) ((sin0 * sin2) + (cos0 * sin1_cos2));
		m[9 + offset] = (float) ((-sin0 * cos2) + (cos0 * sin1_sin2));
		m[10 + offset] = (float) (cos0 * cos1);
	}

	/**
	 * Shows Toast on screen with given message.
	 */
	private void showError(final String errorMsg) {
		new Handler(Looper.getMainLooper()).post(new Runnable() {
			@Override
			public void run() {
				Toast.makeText(mContext, errorMsg, Toast.LENGTH_LONG).show();
			}
		});
	}

}
