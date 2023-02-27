import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.7.21"
    application
    id("me.champeau.jmh") version "0.6.8"
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
    maven {
        url = uri("https://packages.jetbrains.team/maven/p/ki/maven")
    }
}

dependencies {
    api("io.kinference", "ndarray-core", "0.2.8")
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
}

tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "17"
}

java {
    toolchain.languageVersion.set(JavaLanguageVersion.of(17))
}

application {
    mainClass.set("MainKt")
}

jmh {
    warmupIterations.set(5)
    iterations.set(10)
    warmup.set("1s")
    timeOnIteration.set("1s")
    benchmarkMode.addAll("avgt")
    fork.set(2)
    humanOutputFile.set(project.file("${project.buildDir}/results/jmh/human.txt"))
    resultsFile.set(project.file("${project.buildDir}/results/jmh/results.txt"))
    timeUnit.set("us")

    jvmArgsAppend.add("-Xms256m")
    jvmArgsAppend.add("-Xmx8192m")
}
