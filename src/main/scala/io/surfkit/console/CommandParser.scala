package io.surfkit.console

import java.io.File

/**
  * Created by suroot on 22/06/16.
  */
object CommandParser {

  case class Mode(name: String)
  object Mode{
    val none = Mode("none")
    val train = Mode("train")
    val test = Mode("test")
  }

  case class CmdConfig(verbose: Boolean = false, debug: Boolean = false, mode: Mode = Mode.none, files: Seq[File] = Seq(), example: String = "")

  private[this] val parser = new scopt.OptionParser[CmdConfig]("dl4j example") {
    head("dl4j example", "0.0.1-SNAPSHOT")

    opt[Unit]("verbose").action( (_, c) =>
      c.copy(verbose = true) ).text("verbose is a flag")

    opt[Unit]("debug").hidden().action( (_, c) =>
      c.copy(debug = true) ).text("this option is hidden in the usage text")

    help("help").text("prints this usage text")
    note("this utility can be used to alter production data and apply patches.  you must first be ssh port forwarded in order to perform these tasks.\n")

    cmd("train").required().action( (_, c) => c.copy(mode = Mode.train) ).
      text("train is a command.").
      children(
        opt[String]("s").abbr("s").action( (x, c) =>
          c.copy(example = x) ).text("train sequence data")
      )

  }


  def parse(args: Array[String]):Option[CmdConfig] =
  // parser.parse returns Option[C]
    parser.parse(args, CmdConfig())

}